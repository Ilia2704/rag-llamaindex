"""Offline tests: graph, native tool protocol, evidence, repairs, retrieval, tracing."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from agentic.common.trace import Trace
from agentic.retrieval.store import KnowledgeStore
from agentic.multiagent.graph import build_graph, research, structured, validate_finding
from agentic.multiagent.schemas import Finding, Evidence, Plan, Review
from agentic.toolcalling.demo import FinalAnswer

PLAN = {"tasks": [{"role": r, "question": q} for r, q in [
    ("equipment", "Проверь ноутбук и доставку"), ("projects", "Atlas ответственные системы"),
    ("security", "ZTA-17 требования доступа")]]}
GOOD_REVIEW = {"approved": True, "issues": [], "rerun_roles": []}


def tool(name, args, ident="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": ident}])


class Runnable:
    def __init__(self, items):
        self.items = iter(items)
        self.messages = []

    def invoke(self, messages, **kwargs):
        self.messages.append(messages)
        value = next(self.items)
        if isinstance(value, Exception):
            raise value
        return value


class ScriptedLLM:
    def __init__(self, tools=(), structured_values=()):
        self.caller = Runnable(tools)
        self.formatter = Runnable(structured_values)

    def bind_tools(self, definitions):
        self.definitions = definitions
        return self.caller

    def with_structured_output(self, schema, method):
        self.schema = schema
        return self.formatter


class AgenticTests(unittest.TestCase):
    def setUp(self):
        self.stream = io.StringIO()
        self.redirect = contextlib.redirect_stderr(self.stream)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)
        self.trace = Trace()
        self.store = KnowledgeStore()

    def finding(self, role):
        query = "Atlas" if role == "projects" else "ZTA-17"
        category = "Projects" if role == "projects" else "IT_Security"
        doc = self.store.search(query, category)[0]
        return Finding(role=role, status="ok", evidence=[Evidence(document_id=doc["document_id"], quote=doc["text"])])

    def test_real_corpus_filters_and_empty_search(self):
        hits = self.store.search("Atlas", "Projects")
        self.assertTrue(hits)
        self.assertTrue(all(h["category"] == "Projects" and h["year"] == 2026 for h in hits))
        self.assertEqual(self.store.search("невозможныйтокенxyz", "Projects"), [])
        self.assertEqual(self.store.search("Atlas", "Projects", 1900), [])

    def test_document_access_and_paths(self):
        for ident in ("../../.env", "/etc/passwd", "missing.md", "IT_Zero_Trust_2026.md"):
            with self.assertRaises(ValueError):
                self.store.read(ident, "Projects")

    def test_empty_corpus(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "пуст"):
                KnowledgeStore(root)

    def test_symlink_not_read(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "leak.md").symlink_to(Path(__file__).resolve())
            with self.assertRaises(ValueError):
                KnowledgeStore(root)

    def test_qdrant_not_initialized_locally(self):
        self.store.search("Atlas", "Projects")
        self.assertIsNone(self.store.index)

    def test_research_native_calls_and_read(self):
        finding = self.finding("projects")
        ident = finding.evidence[0].document_id
        llm = ScriptedLLM([
            tool("search_knowledge_base", {"query": "Atlas", "category": "Projects"}),
            tool("read_document", {"document_id": ident}, "2"), AIMessage(content="Готово")],
            [finding.model_dump(mode="json")])
        result = research("projects", "Кто отвечает за Atlas?", llm, self.store, self.trace)
        self.assertEqual(result, finding)
        self.assertIn("TOOL_RESULT_VALID", self.stream.getvalue())
        self.assertEqual(len(llm.caller.messages), 3)

    def test_research_empty_evidence(self):
        llm = ScriptedLLM([tool("search_knowledge_base", {"query": "xyz", "category": "Projects"}),
                          AIMessage(content="Нет данных")],
                         [{"role": "projects", "status": "insufficient", "evidence": []}])
        self.assertEqual(research("projects", "xyz", llm, self.store, self.trace).status, "insufficient")

    def test_research_injected_repair(self):
        finding = self.finding("security")
        args = {"query": "ZTA-17", "category": "IT_Security"}
        llm = ScriptedLLM([tool("search_knowledge_base", args), tool("search_knowledge_base", args, "2"),
                          AIMessage(content="Готово")], [finding.model_dump(mode="json")])
        self.assertEqual(research("security", "Доступ", llm, self.store, self.trace, inject=True), finding)
        self.assertIn("REPAIR_SUCCEEDED", self.stream.getvalue())

    def test_wrong_category_is_blocked_and_repaired(self):
        finding = self.finding("projects")
        llm = ScriptedLLM([tool("search_knowledge_base", {"query": "Atlas", "category": "IT_Security"}),
                          tool("search_knowledge_base", {"query": "Atlas", "category": "Projects"}, "2"),
                          AIMessage(content="Готово")], [finding.model_dump(mode="json")])
        research("projects", "Atlas", llm, self.store, self.trace)
        self.assertIn("доступна только", self.stream.getvalue())

    def test_forbidden_tool_repair_exhausted(self):
        llm = ScriptedLLM([tool("shell", {}), tool("shell", {}, "2")])
        with self.assertRaisesRegex(ValueError, "Лимит исправлений"):
            research("projects", "Atlas", llm, self.store, self.trace)

    def test_read_before_search_blocked(self):
        llm = ScriptedLLM([tool("read_document", {"document_id": "Project_Atlas_Graph_2026.md"})])
        with self.assertRaisesRegex(ValueError, "Лимит шагов"):
            research("projects", "Atlas", llm, self.store, self.trace, max_steps=1)
        self.assertIn("Сначала найди", self.stream.getvalue())

    def test_no_tool_call(self):
        with self.assertRaisesRegex(ValueError, "не выполнил поиск"):
            research("projects", "Atlas", ScriptedLLM([AIMessage(content="Ответ из памяти")]), self.store, self.trace)

    def test_malformed_call(self):
        response = AIMessage(content="", invalid_tool_calls=[{"name": "search", "args": "{bad", "id": "1", "error": "bad"}])
        with self.assertRaisesRegex(ValueError, "Непарсируемый"):
            research("projects", "Atlas", ScriptedLLM([response]), self.store, self.trace)

    def test_too_many_calls(self):
        response = AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": str(i)} for i in range(4)])
        with self.assertRaisesRegex(ValueError, "Слишком много"):
            research("projects", "Atlas", ScriptedLLM([response]), self.store, self.trace)

    def test_forged_evidence_and_role(self):
        finding = self.finding("projects")
        for role, seen in [("security", {}), ("projects", {}),
                           ("projects", {finding.evidence[0].document_id: "Другой текст"})]:
            with self.assertRaises(ValueError):
                validate_finding(finding, role, seen)

    def test_missing_and_contradictory_evidence(self):
        with self.assertRaises(ValueError):
            validate_finding(Finding(role="projects", status="ok", evidence=[]), "projects", {})
        finding = self.finding("projects").model_copy(update={"status": "insufficient"})
        with self.assertRaises(ValueError):
            validate_finding(finding, "projects", {})

    def test_schema_repair_and_limit(self):
        llm = ScriptedLLM(structured_values=[{"bad": 1}, GOOD_REVIEW])
        review = structured(llm, Review, [], self.trace, "TEST")
        self.assertTrue(review.approved)
        llm = ScriptedLLM(structured_values=[{"bad": 1}, {"bad": 2}])
        with self.assertRaises(ValidationError):
            structured(llm, Review, [], self.trace, "TEST")

    def test_network_failure_propagates(self):
        with self.assertRaises(TimeoutError):
            research("projects", "Atlas", ScriptedLLM([TimeoutError("timeout")]), self.store, self.trace)

    def execute_graph(self, reviews=None, **kwargs):
        llm = ScriptedLLM(structured_values=[PLAN, *(reviews or [GOOD_REVIEW])])
        equipment = FinalAnswer(order_id=271, order_status="processing", delivery_status=None, estimated_delivery=None)
        with patch("agentic.multiagent.graph.equipment_run", return_value=equipment) as equipment_mock, \
             patch("agentic.multiagent.graph.research", side_effect=lambda role, *a, **k: self.finding(role)) as research_mock:
            state = build_graph(llm, self.store, self.trace, **kwargs).invoke({"question": "Онбординг Atlas"})
        return state, equipment_mock, research_mock

    def test_graph_success(self):
        state, equipment, workers = self.execute_graph()
        self.assertFalse(state["answer"].access_verified)
        self.assertEqual(len(state["answer"].findings), 2)
        self.assertEqual(equipment.call_count, 1)
        self.assertEqual(workers.call_count, 2)

    def test_injected_missing_result_is_rechecked(self):
        state, equipment, workers = self.execute_graph([GOOD_REVIEW, GOOD_REVIEW], inject_critic_error=True)
        self.assertEqual(state["revisions"], 1)
        self.assertEqual(equipment.call_count, 1)
        self.assertEqual(workers.call_count, 3)
        self.assertEqual(workers.call_args.args[0], "security")

    def test_critic_targeted_revision(self):
        bad = {"approved": False, "issues": ["Уточни требования доступа"], "rerun_roles": ["security"]}
        state, _, workers = self.execute_graph([bad, GOOD_REVIEW])
        self.assertEqual(state["revisions"], 1)
        self.assertEqual(workers.call_count, 3)

    def test_critic_not_approved_after_revision(self):
        bad = {"approved": False, "issues": ["Проблема"], "rerun_roles": ["security"]}
        with self.assertRaisesRegex(ValueError, "не одобрил"):
            self.execute_graph([bad, bad])

    def test_critic_no_action_fails(self):
        bad = {"approved": False, "issues": ["Проблема"], "rerun_roles": []}
        with self.assertRaisesRegex(ValueError, "не одобрил"):
            self.execute_graph([bad])

    def test_invalid_plan_is_repaired(self):
        llm = ScriptedLLM(structured_values=[{"tasks": [PLAN["tasks"][0]]}, PLAN, GOOD_REVIEW])
        equipment = FinalAnswer(order_id=271, order_status="processing", delivery_status=None, estimated_delivery=None)
        with patch("agentic.multiagent.graph.equipment_run", return_value=equipment), \
             patch("agentic.multiagent.graph.research", side_effect=lambda role, *a, **k: self.finding(role)):
            build_graph(llm, self.store, self.trace).invoke({"question": "Atlas"})
        self.assertIn("PLAN_ERROR", self.stream.getvalue())

    def test_full_graph_with_real_tools_all_orders_and_injections(self):
        # No worker functions are mocked: run the entire LangGraph/tool/validation path.
        for order_id, status, inject, critic_error in [
            (271, "shipped", False, False), (271, "shipped", True, False),
            (271, "shipped", True, True), (272, "processing", False, False),
            (999, "not_found", False, False),
        ]:
            with self.subTest(order_id=order_id, inject=inject, critic_error=critic_error):
                calls = []
                if inject:
                    calls.append(tool("get_order", {"order_id": order_id}, "bad"))
                calls.append(tool("get_order", {"order_id": order_id}, "order"))
                equipment = {"order_id": order_id, "order_status": status,
                             "delivery_status": None, "estimated_delivery": None}
                if status == "shipped":
                    calls.append(tool("get_delivery", {"shipment_id": "SHP-271"}, "delivery"))
                    equipment.update(delivery_status="in_transit", estimated_delivery="2026-10-15")
                finals = [PLAN, equipment]
                for role in ["projects", "security"]:
                    calls.extend([tool("search_knowledge_base", {
                        "query": "Atlas" if role == "projects" else "ZTA-17",
                        "category": "Projects" if role == "projects" else "IT_Security"}, role),
                        AIMessage(content="Готово")])
                    finals.append(self.finding(role).model_dump(mode="json"))
                finals.append(GOOD_REVIEW)
                if critic_error:
                    calls.extend([tool("search_knowledge_base", {"query": "ZTA-17", "category": "IT_Security"}, "revision"),
                                  AIMessage(content="Готово")])
                    finals.extend([self.finding("security").model_dump(mode="json"), GOOD_REVIEW])
                llm = ScriptedLLM(calls, finals)
                state = build_graph(llm, self.store, self.trace, order_id=order_id,
                                    inject_invalid_args=inject, inject_critic_error=critic_error).invoke({"question": "Atlas"})
                self.assertEqual(state["answer"].equipment.order_status, status)
                self.assertEqual(state["revisions"], int(critic_error))
                self.assertEqual(len(state["answer"].findings), 2)
                self.assertFalse(state["answer"].access_verified)

    def test_malformed_structured_output_repaired(self):
        llm = ScriptedLLM(structured_values=[ValueError("Invalid JSON"), GOOD_REVIEW])
        self.assertTrue(structured(llm, Review, [], self.trace, "JSON").approved)

    def test_forged_retrieval_result_blocked(self):
        with patch.object(self.store, "search", return_value=[{
            "document_id": "Project_Atlas_Graph_2026.md", "year": 2026,
            "text": "Выдуманный текст", "category": "Projects", "score": 1.0}]):
            llm = ScriptedLLM([tool("search_knowledge_base", {"query": "Atlas", "category": "Projects"})])
            with self.assertRaisesRegex(ValueError, "Лимит шагов"):
                research("projects", "Atlas", llm, self.store, self.trace, max_steps=1)
        self.assertIn("Неверный результат retrieval", self.stream.getvalue())

    def test_duplicate_plan_roles_rejected(self):
        bad = {"tasks": [PLAN["tasks"][0]] * 3}
        llm = ScriptedLLM(structured_values=[bad, bad])
        with self.assertRaisesRegex(ValueError, "Повторяющиеся"):
            build_graph(llm, self.store, self.trace).invoke({"question": "Atlas"})

    def test_contradictory_critic_repaired(self):
        bad = {"approved": True, "issues": ["Проблема"], "rerun_roles": []}
        self.execute_graph([bad, GOOD_REVIEW])
        self.assertIn("CRITIC_ERROR", self.stream.getvalue())

    def test_langfuse_context_failure_falls_back(self):
        client = Mock()
        client.start_as_current_observation.return_value.__enter__ = Mock(side_effect=RuntimeError("SDK error"))
        client.start_as_current_observation.return_value.__exit__ = Mock(return_value=False)
        with patch.dict("os.environ", {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}), \
             patch("agentic.common.trace.urlopen") as health, \
             patch("langfuse.Langfuse", return_value=client), \
             patch("langfuse.langchain.CallbackHandler"):
            health.return_value.__enter__.return_value.status = 200
            with self.trace.session("demo", "question", enabled=True):
                self.trace.emit("STILL_WORKS")
        self.assertIn("LANGFUSE_SKIPPED", self.stream.getvalue())

    def test_qdrant_adapter_with_memory_db_and_mock_embeddings(self):
        from llama_index.core import Document, VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        client = QdrantClient(":memory:")
        self.addCleanup(client.close)
        embedding = MockEmbedding(embed_dim=8)
        store = QdrantVectorStore(client=client, collection_name="agentic_test")
        index = VectorStoreIndex.from_vector_store(store, embed_model=embedding)
        docs = []
        for ident, category, year in [("Project_Atlas_Graph_2026.md", "Projects", 2026),
                                       ("IT_Zero_Trust_2026.md", "IT_Security", 2026)]:
            doc = self.store.read(ident, category)
            docs.append(Document(text=doc["text"], metadata={"filename": ident, "category": category, "year": year}))
        for doc in docs:
            index.insert(doc)
        adapter = KnowledgeStore(backend="qdrant")
        with patch.dict("os.environ", {"AGENTIC_COLLECTION": "agentic_test"}), \
             patch("qdrant_client.QdrantClient", return_value=client), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding", return_value=embedding):
            hits = adapter.search("Atlas", "Projects")
            self.assertTrue(hits)
            self.assertTrue(all(item["category"] == "Projects" for item in hits))
            self.assertEqual(adapter.search("Atlas", "Projects", 1900), [])
            missing = KnowledgeStore(backend="qdrant")
            with patch.dict("os.environ", {"AGENTIC_COLLECTION": "absent"}):
                with self.assertRaisesRegex(ValueError, "Нет RAG-индекса"):
                    missing.search("Atlas", "Projects")

    def test_cli_failure_and_no_langfuse(self):
        from agentic.multiagent.demo import main
        with patch("sys.argv", ["demo"]), \
             patch("agentic.multiagent.demo.ChatOllama", side_effect=TimeoutError("offline")), \
             patch("agentic.common.trace.urlopen") as network:
            self.assertEqual(main(), 1)
            network.assert_not_called()
        self.assertIn("FAILED", self.stream.getvalue())

    def test_cli_invalid_order(self):
        from agentic.multiagent.demo import main
        with patch("sys.argv", ["demo", "--order-id", "-1"]):
            with self.assertRaises(SystemExit) as exit:
                main()
        self.assertEqual(exit.exception.code, 2)

    def test_tracing_disabled_no_network(self):
        with patch("agentic.common.trace.urlopen") as network:
            with self.trace.session("demo", "question"):
                self.trace.emit("EVENT", {"ok": True})
            network.assert_not_called()

    def test_langfuse_unavailable_falls_back(self):
        with patch.dict("os.environ", {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}), \
             patch("agentic.common.trace.urlopen", side_effect=OSError("offline")):
            with self.trace.session("demo", "question", enabled=True):
                self.trace.emit("STILL_WORKS")
        self.assertIn("LANGFUSE_SKIPPED", self.stream.getvalue())

    def test_langfuse_distinct_root_names_and_events(self):
        root = Mock()
        context = Mock()
        context.__enter__ = Mock(return_value=root)
        context.__exit__ = Mock(return_value=False)
        client = Mock()
        client.start_as_current_observation.return_value = context
        with patch.dict("os.environ", {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}), \
             patch("agentic.common.trace.urlopen") as health, \
             patch("langfuse.Langfuse", return_value=client), \
             patch("langfuse.langchain.CallbackHandler"):
            health.return_value.__enter__.return_value.status = 200
            for name in ("agentic-atlas-toolcalling", "agentic-atlas-multiagent"):
                with self.trace.session(name, "question", enabled=True):
                    self.trace.emit("TOOL_CALL", {"x": 1})
            names = [call.kwargs["name"] for call in client.start_as_current_observation.call_args_list]
            self.assertIn("agentic-atlas-toolcalling", names)
            self.assertIn("agentic-atlas-multiagent", names)
            self.assertEqual(client.flush.call_count, 2)


if __name__ == "__main__":
    unittest.main()
