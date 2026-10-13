"""Offline orchestration checks: no Ollama server or model required."""
import contextlib
import io
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from agentic.toolcalling.demo import (
    FinalAnswer, OrderArgs, TOOLS, ToolSpec, run,
)


def call(name, args, ident="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": ident}])


# Parse ISO dates using the same strict JSON boundary as production.
ANSWER = FinalAnswer.model_validate_json(
    '{"order_id":271,"order_status":"shipped","delivery_status":"in_transit",'
    '"estimated_delivery":"2026-10-15"}')


class FakeModel:
    def __init__(self, replies, finals=None):
        self.replies = iter(replies)
        self.finals = iter(finals or [ANSWER])
        self.formatting = False

    def bind_tools(self, definitions):
        assert len(definitions) == 2
        return self

    def with_structured_output(self, schema, method):
        assert schema == FinalAnswer.model_json_schema() and method == "json_schema"
        self.formatting = True
        return self

    def invoke(self, messages):
        value = next(self.finals if self.formatting else self.replies)
        return value.model_dump(mode="json") if isinstance(value, FinalAnswer) else value


class ToolCallingTests(unittest.TestCase):
    def execute(self, model, **kwargs):
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            result = run("Где заказ 271?", model, **kwargs)
        return result, output.getvalue()

    def test_success_and_vertical_trace(self):
        result, trace = self.execute(FakeModel([
            call("get_order", {"order_id": 271}),
            call("get_delivery", {"shipment_id": "SHP-271"}, "2")]))
        self.assertEqual(result, ANSWER)
        events = [line for line in trace.splitlines() if line.startswith("[")]
        self.assertTrue(any("ARGUMENTS_VALID" in line for line in events))
        self.assertTrue(any("TOOL_RESULT_SENT" in line for line in events))
        self.assertIn("FINAL_VALID", events[-1])

    def test_strict_arguments(self):
        for args in ({"order_id": -1}, {"order_id": "271"}, {"order_id": True},
                     {"order_id": 271, "extra": 1}):
            with self.assertRaises(ValidationError):
                OrderArgs.model_validate(args)

    def test_unknown_order_and_processing(self):
        for ident, status in ((999, "not_found"), (272, "processing")):
            answer = FinalAnswer(order_id=ident, order_status=status,
                                 delivery_status=None, estimated_delivery=None)
            result, _ = self.execute(FakeModel([call("get_order", {"order_id": ident})], [answer]))
            self.assertEqual(result, answer)

    def test_injected_error_is_repaired(self):
        result, trace = self.execute(FakeModel([
            call("get_order", {"order_id": 271}),
            call("get_order", {"order_id": 271}, "2"),
            call("get_delivery", {"shipment_id": "SHP-271"}, "3")]), inject_invalid_args=True)
        self.assertEqual(result, ANSWER)
        self.assertIn("DEMO_INJECT_INVALID_ARGS", trace)
        self.assertIn("REPAIR_REQUESTED", trace)
        self.assertIn("REPAIR_SUCCEEDED", trace)
        self.assertIn('"order_id": -1', trace)

    def test_unknown_tool_repair_limit(self):
        with self.assertRaisesRegex(ValueError, "лимит исправлений"):
            self.execute(FakeModel([call("evil", {}), call("evil", {}, "2")]))

    def test_wrong_shipment_not_executed(self):
        spec = TOOLS["get_delivery"]
        from unittest.mock import Mock
        function = Mock()
        blocked = ToolSpec(spec.args, spec.result, function, spec.description)
        with patch.dict(TOOLS, {"get_delivery": blocked}):
            with self.assertRaisesRegex(ValueError, "лимит шагов"):
                self.execute(FakeModel([call("get_delivery", {"shipment_id": "SHP-999"})]), max_steps=1)
            function.assert_not_called()

    def test_invalid_tool_result(self):
        spec = TOOLS["get_order"]
        invalid = ToolSpec(spec.args, spec.result, lambda **_: {"unexpected": True}, spec.description)
        with patch.dict(TOOLS, {"get_order": invalid}):
            with self.assertRaisesRegex(ValueError, "лимит шагов"):
                self.execute(FakeModel([call("get_order", {"order_id": 271})]), max_steps=1)

    def test_missing_calls_fails(self):
        with self.assertRaisesRegex(ValueError, "get_order"):
            self.execute(FakeModel([AIMessage(content="Я думаю, заказ уже доставлен")]))

    def test_final_facts_repair(self):
        wrong = ANSWER.model_copy(update={"order_id": 999})
        result, trace = self.execute(FakeModel([
            call("get_order", {"order_id": 271}),
            call("get_delivery", {"shipment_id": "SHP-271"}, "2")], [wrong, ANSWER]))
        self.assertEqual(result, ANSWER)
        self.assertIn("FINAL_VALIDATION_ERROR", trace)
        self.assertIn("FINAL_REPAIR_REQUESTED", trace)
        self.assertIn("FINAL_REPAIR_SUCCEEDED", trace)

    def test_final_facts_retry_exhausted(self):
        wrong = ANSWER.model_copy(update={"order_id": 999})
        with self.assertRaisesRegex(ValueError, "не совпадают"):
            self.execute(FakeModel([
                call("get_order", {"order_id": 271}),
                call("get_delivery", {"shipment_id": "SHP-271"}, "2")], [wrong, wrong]))


if __name__ == "__main__":
    unittest.main()
