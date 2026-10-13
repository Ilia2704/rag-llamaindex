"""Two read-only tools, bounded repair, and a validated structured answer."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Callable, Literal

from agentic.common.trace import Trace
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class OrderArgs(Contract):
    order_id: Annotated[int, Field(gt=0)]


class DeliveryArgs(Contract):
    shipment_id: Annotated[str, Field(pattern=r"^SHP-\d+$")]


class OrderResult(Contract):
    order_id: Annotated[int, Field(gt=0)]
    found: bool
    order_status: Literal["shipped", "processing"] | None
    shipment_id: Annotated[str, Field(pattern=r"^SHP-\d+$")] | None


    @model_validator(mode="after")
    def consistent(self):
        if not self.found and (self.order_status is not None or self.shipment_id is not None):
            raise ValueError("Отсутствующий заказ не имеет статуса или отправления")
        if self.found and self.order_status is None:
            raise ValueError("Найденный заказ требует статуса")
        if self.order_status == "shipped" and self.shipment_id is None:
            raise ValueError("Отправленный заказ требует shipment_id")
        if self.order_status == "processing" and self.shipment_id is not None:
            raise ValueError("Обрабатываемый заказ еще не отправлен")
        return self


class DeliveryResult(Contract):
    shipment_id: Annotated[str, Field(pattern=r"^SHP-\d+$")]
    found: bool
    delivery_status: Literal["in_transit", "delivered"] | None
    estimated_delivery: date | None


    @model_validator(mode="after")
    def consistent(self):
        if not self.found and (self.delivery_status is not None or self.estimated_delivery is not None):
            raise ValueError("Отсутствующая доставка не имеет статуса или даты")
        if self.found and self.delivery_status is None:
            raise ValueError("Найденная доставка требует статуса")
        return self


class FinalAnswer(Contract):
    order_id: Annotated[int, Field(gt=0)]
    order_status: Literal["shipped", "processing", "not_found"]
    delivery_status: Literal["in_transit", "delivered", "not_found"] | None
    estimated_delivery: date | None


def get_order(order_id: int) -> dict:
    """Demo Atlas laptop orders: 271 shipped, 272 processing."""
    data = {271: ("shipped", "SHP-271"), 272: ("processing", None)}
    status, shipment = data.get(order_id, (None, None))
    return dict(order_id=order_id, found=order_id in data,
                order_status=status, shipment_id=shipment)


def get_delivery(shipment_id: str) -> dict:
    """Look up the delivery status and estimated date for a shipment."""
    found = shipment_id == "SHP-271"
    return dict(shipment_id=shipment_id, found=found,
                delivery_status="in_transit" if found else None,
                estimated_delivery="2026-10-15" if found else None)


@dataclass(frozen=True)
class ToolSpec:
    args: type[Contract]
    result: type[Contract]
    function: Callable
    description: str

    def definition(self, name: str) -> dict:
        return {"type": "function", "function": {
            "name": name, "description": self.description,
            "parameters": self.args.model_json_schema(),
        }}


TOOLS = {
    "get_order": ToolSpec(OrderArgs, OrderResult, get_order,
                         "Получить статус заказа и shipment_id по order_id."),
    "get_delivery": ToolSpec(DeliveryArgs, DeliveryResult, get_delivery,
                            "Получить доставку по shipment_id из результата get_order."),
}



def validate_result(model: type[Contract], raw: dict) -> Contract:
    # JSON mode supports ISO dates while retaining strict numeric/bool validation.
    return model.model_validate_json(json.dumps(raw))


def expected_answer(results: dict[str, Contract]) -> FinalAnswer:
    order = results.get("get_order")
    if not isinstance(order, OrderResult):
        raise ValueError("Сначала требуется успешный get_order.")
    status = order.order_status if order.found else "not_found"
    delivery_status = None
    estimated = None
    if order.shipment_id is not None:
        delivery = results.get("get_delivery")
        if not isinstance(delivery, DeliveryResult) or delivery.shipment_id != order.shipment_id:
            raise ValueError("Требуется get_delivery для shipment_id этого заказа.")
        delivery_status = delivery.delivery_status if delivery.found else "not_found"
        estimated = delivery.estimated_delivery
    return FinalAnswer(order_id=order.order_id, order_status=status,
                       delivery_status=delivery_status, estimated_delivery=estimated)


SYSTEM = """Ты демонстрационный помощник по заказам. Получи факты только через инструменты.
Сначала вызови get_order для номера из запроса. Если shipment_id не null,
вызови get_delivery с этим shipment_id. Для отсутствующего заказа или заказа
без отправления get_delivery не нужен. Не придумывай идентификаторы и даты.
При ошибке исправь аргументы. Не повторяй успешно выполненные вызовы.
После получения всех данных закончи вызовы инструментов.
"""


def run(question: str, llm, *, max_steps: int = 5,
        inject_invalid_args: bool = False, trace: Trace | None = None) -> FinalAnswer:
    trace = trace or Trace()
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=question)]
    caller = llm.bind_tools([spec.definition(name) for name, spec in TOOLS.items()])
    results: dict[str, Contract] = {}
    errors = 0
    injected = False
    failed_call = None
    trace.emit("USER", question)
    for step in range(1, max_steps + 1):
        trace.emit("LLM_REQUEST", {"step": step})
        reply = trace.invoke(caller, messages)
        messages.append(reply)
        if reply.invalid_tool_calls:
            raise ValueError("Модель вернула непарсируемый tool call.")
        calls = reply.tool_calls
        trace.emit("LLM_TOOL_CALLS", calls)
        if not calls:
            # Missing evidence is an error; do not silently invent or force calls.
            expected_answer(results)
            break
        if len(calls) > 3:
            raise ValueError("Слишком много вызовов за шаг")
        for call in calls:
            name, args = call["name"], call["args"]
            trace.emit("TOOL_CALL", {"name": name, "arguments": args})
            try:
                if name not in TOOLS:
                    raise ValueError(f"Инструмент {name!r} отсутствует в разрешенном реестре.")
                spec = TOOLS[name]
                if inject_invalid_args and not injected and name == "get_order":
                    args = {"order_id": -1}
                    injected = True
                    trace.emit("DEMO_INJECT_INVALID_ARGS", args)
                validated = spec.args.model_validate(args)
                trace.emit("ARGUMENTS_VALID", validated.model_dump(mode="json"))
                if name in results:
                    raise ValueError("Успешный инструмент уже выполнен; повтор запрещен.")
                if name == "get_delivery":
                    order = results.get("get_order")
                    if not isinstance(order, OrderResult) or not order.found or order.shipment_id != validated.shipment_id:
                        raise ValueError("shipment_id должен принадлежать найденному заказу.")
                trace.emit("TOOL_EXECUTE", name)
                result = validate_result(spec.result, spec.function(**validated.model_dump()))
                if name == "get_order" and result.order_id != validated.order_id:
                    raise ValueError("Результат инструмента относится к другому заказу")
                if name == "get_delivery" and result.shipment_id != validated.shipment_id:
                    raise ValueError("Результат инструмента относится к другой доставке")
                if failed_call is not None:
                    trace.emit("REPAIR_SUCCEEDED", {"previous": failed_call,
                               "corrected": {"name": name, "arguments": args}})
                    failed_call = None
                results[name] = result
                payload = result.model_dump(mode="json")
                trace.emit("RESULT_VALID", payload)
                messages.append(ToolMessage(content=json.dumps(payload), tool_call_id=call["id"], name=name))
                trace.emit("TOOL_RESULT_SENT", name)
            except (ValidationError, ValueError) as exc:
                errors += 1
                trace.emit("VALIDATION_ERROR", str(exc))
                if errors > 1:
                    raise ValueError("Исчерпан лимит исправлений инструментов.") from exc
                failed_call = {"name": name, "arguments": args}
                feedback = {"error": str(exc), "failed_call": failed_call,
                            "instruction": "Повтори вызов с исправленными аргументами из запроса пользователя."}
                messages.append(ToolMessage(content=json.dumps(feedback, ensure_ascii=False),
                                            tool_call_id=call["id"], name=name, status="error"))
                trace.emit("REPAIR_REQUESTED", feedback)
        if failed_call is not None:
            continue
        try:
            expected_answer(results)
        except ValueError:
            continue
        break
    else:
        raise ValueError("Исчерпан лимит шагов вызова инструментов.")

    expected = expected_answer(results)
    # Separate structured generation from native tool selection.
    formatter = llm.with_structured_output(FinalAnswer.model_json_schema(), method="json_schema")
    final_messages = [SystemMessage(content="Верни JSON по схеме. Точно перенеси факты инструментов. "
                                   "Отсутствующий заказ: order_status=not_found. "
                                   "Без отправления: delivery_status и estimated_delivery=null."),
                      HumanMessage(content=json.dumps({name: result.model_dump(mode="json")
                                                       for name, result in results.items()}, ensure_ascii=False))]
    for attempt in range(2):
        trace.emit("STRUCTURED_OUTPUT_REQUEST", {"attempt": attempt + 1})
        raw_answer = None
        try:
            raw_answer = trace.invoke(formatter, final_messages)
            trace.emit("FINAL_CANDIDATE", raw_answer)
            answer = validate_result(FinalAnswer, raw_answer)
            if answer != expected:
                raise ValueError("Поля финального ответа не совпадают с результатами инструментов.")
            if attempt:
                trace.emit("FINAL_REPAIR_SUCCEEDED", answer.model_dump(mode="json"))
            trace.emit("FINAL_VALID", answer.model_dump(mode="json"))
            return answer
        except (ValidationError, ValueError) as exc:
            trace.emit("FINAL_VALIDATION_ERROR", str(exc))
            if attempt:
                raise
            feedback = {"error": str(exc), "previous_answer": raw_answer,
                        "instruction": "Исправь ответ, используя исходные факты инструментов."}
            trace.emit("FINAL_REPAIR_REQUESTED", feedback)
            final_messages.append(HumanMessage(content=json.dumps(feedback, ensure_ascii=False)))
    raise RuntimeError("Не получен финальный ответ.")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="Где мой ноутбук для проекта Atlas, заказ №271, и когда доставка?")
    parser.add_argument("--model", default=os.getenv("TOOLCALLING_MODEL", "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"))
    parser.add_argument("--base-url", default=os.getenv("TOOLCALLING_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-steps", type=int, choices=range(1, 11), default=5)
    parser.add_argument("--inject-invalid-args", action="store_true")
    parser.add_argument("--langfuse", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout должен быть положительным")
    trace = Trace()
    trace.emit("CONFIG", {"model": args.model, "base_url": args.base_url,
                          "timeout": args.timeout, "max_steps": args.max_steps})
    try:
        llm = ChatOllama(model=args.model, base_url=args.base_url, temperature=0,
                         reasoning=False, client_kwargs={"timeout": args.timeout})
        with trace.session("agentic-atlas-toolcalling", args.question, args.langfuse):
            answer = run(args.question, llm, max_steps=args.max_steps,
                         inject_invalid_args=args.inject_invalid_args, trace=trace)
        print(answer.model_dump_json(indent=2))
        return 0
    except Exception as exc:
        trace.emit("FAILED", str(exc))
        if getattr(exc, "status_code", None) == 404:
            trace.emit("MODEL_SETUP", {
                "message": "Проверьте точное имя установленной модели через ollama list. "
                           "Передайте его через --model или скачайте выбранную модель.",
                "list_command": "ollama list",
                "pull_command": f"ollama pull {args.model}",
            })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
