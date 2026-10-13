"""Atlas onboarding with specialized agents and optional Langfuse."""
import argparse
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from agentic.common.trace import Trace
from agentic.retrieval.store import KnowledgeStore
from .graph import build_graph

QUESTION = "Я выхожу в команду Atlas. Проверь заказ ноутбука, дату доставки, ответственных за проект, системы и требования доступа."


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=QUESTION)
    parser.add_argument("--order-id", type=int, default=271)
    parser.add_argument("--model", default=os.getenv("AGENTIC_MODEL", "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"))
    parser.add_argument("--base-url", default=os.getenv("AGENTIC_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retrieval", choices=["local", "qdrant"], default="local")
    parser.add_argument("--inject-invalid-args", action="store_true")
    parser.add_argument("--inject-critic-error", action="store_true")
    parser.add_argument("--langfuse", action="store_true")
    args = parser.parse_args()
    if args.order_id <= 0 or args.timeout <= 0:
        parser.error("order-id и timeout должны быть положительными")
    trace = Trace()
    try:
        with trace.session("agentic-atlas-multiagent", args.question, args.langfuse):
            trace.emit("CONFIG", vars(args))
            llm = ChatOllama(model=args.model, base_url=args.base_url, temperature=0,
                             reasoning=False, client_kwargs={"timeout": args.timeout})
            app = build_graph(llm, KnowledgeStore(backend=args.retrieval), trace, order_id=args.order_id,
                              inject_invalid_args=args.inject_invalid_args,
                              inject_critic_error=args.inject_critic_error)
            state = app.invoke({"question": args.question}, config={"recursion_limit": 12})
            print(state["answer"].model_dump_json(indent=2))
        return 0
    except Exception as exc:
        trace.emit("FAILED", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
