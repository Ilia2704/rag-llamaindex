"""LlamaIndex RAG demo over the existing RoboTech markdown documents."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import requests
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "knowledge_base"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "robotex_docs_llamaindex"
LLM_MODEL = "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

QA_TEMPLATE = """\
Ты — корпоративный ассистент компании "РобоТех".
Ответь на вопрос сотрудника, используя ТОЛЬКО предоставленный ниже контекст.
Если в контексте нет информации, скажи "В документах нет информации об этом".
Не придумывай факты.

Контекст:
---------------------
{context_str}
---------------------

Вопрос: {query_str}

Ответ:
"""


def select_device() -> str:
    """Choose the best local accelerator for HuggingFace embeddings."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def infer_metadata(path: Path, content: str) -> dict[str, object]:
    years = re.findall(r"202[4-6]", content)
    filename = path.name

    if "HR" in filename or "Holiday" in filename:
        category = "HR"
    elif "Security" in filename or "VPN" in filename:
        category = "IT_Security"
    elif "Project" in filename:
        category = "Projects"
    else:
        category = "General"

    return {
        "source": str(path.relative_to(PROJECT_ROOT)),
        "filename": filename,
        "year": int(years[0]) if years else 2026,
        "category": category,
    }


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Не найдена папка {data_dir}. Сначала выполните: "
            "uv run scripts/generate_data.py"
        )

    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        documents.append(
            # Document: исходный файл + metadata до разбиения на ноды.
            Document(
                text=content,
                metadata=infer_metadata(path, content),
                id_=str(path.relative_to(PROJECT_ROOT)),
            )
        )

    if not documents:
        raise FileNotFoundError(f"В {data_dir} нет markdown-файлов.")

    return documents


def check_ollama(model: str = LLM_MODEL) -> bool:
    print("Проверка Ollama...", end=" ")
    try:
        response = requests.get(f"{OLLAMA_URL}/", timeout=10)
        response.raise_for_status()
        print("OK")
    except Exception as exc:
        print(f"ошибка: {exc}")
        return False

    print(f"Проверка модели {model}...", end=" ")
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        response.raise_for_status()
        models = [item["name"] for item in response.json().get("models", [])]
    except Exception as exc:
        print(f"ошибка: {exc}")
        return False

    if any(model in name for name in models):
        print("найдена")
        return True

    print(f"не найдена. Доступные модели: {models}")
    print(f"Выполните: ollama pull {model}")
    return False


def check_qdrant() -> bool:
    print("Проверка Qdrant...", end=" ")
    try:
        response = requests.get(f"{QDRANT_URL}/collections", timeout=10)
        response.raise_for_status()
        count = len(response.json()["result"]["collections"])
        print(f"OK, коллекций: {count}")
        return True
    except Exception as exc:
        print(f"ошибка: {exc}")
        return False


def configure_settings(model: str = LLM_MODEL) -> None:
    device = select_device()
    print(f"Embedding model: {EMBEDDING_MODEL} on {device}")
    print(f"LLM via LlamaIndex: {model} at {OLLAMA_URL}")

    # Settings задает основные LlamaIndex-компоненты для текущего процесса.
    Settings.llm = Ollama(
        model=model,
        base_url=OLLAMA_URL,
        temperature=0.1,
        request_timeout=180.0,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device=device,
        normalize=True,
    )
    # Node parser превращает Document в Node-чанки.
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    return client.collection_exists(collection_name)


def build_index(
    *,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
    model: str = LLM_MODEL,
) -> VectorStoreIndex | None:
    configure_settings(model)
    client = qdrant_client()

    if collection_exists(client, collection_name):
        if not reset:
            print(
                f"Коллекция {collection_name!r} уже существует. "
                "Индексация пропущена, чтобы не задублировать чанки. "
                "Для пересборки выполните команду index с --reset."
            )
            return None
        print(f"Удаляю и пересоздаю коллекцию {collection_name!r} в Qdrant...")
        client.delete_collection(collection_name)

    documents = load_documents()
    print(f"Загружено документов: {len(documents)}")

    # QdrantVectorStore: LlamaIndex-адаптер к Qdrant collection.
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )
    # StorageContext связывает индекс с выбранным vector store.
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # VectorStoreIndex режет Documents на Nodes, считает embeddings и пишет в Qdrant.
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"Индекс готов: Qdrant collection={collection_name!r}")
    return index


def format_relationships(node: object) -> str:
    relationships = getattr(node, "relationships", {})
    if not relationships:
        return "none"

    parts: list[str] = []
    for relation_type, relation_info in relationships.items():
        relation_name = getattr(relation_type, "name", str(relation_type))
        if isinstance(relation_info, list):
            target_ids = [
                getattr(item, "node_id", str(item))
                for item in relation_info
            ]
            relation_target = ", ".join(target_ids)
        else:
            relation_target = getattr(relation_info, "node_id", str(relation_info))
        parts.append(f"{relation_name}->{relation_target}")
    return "; ".join(parts)


def show_nodes(limit: int = 10, chars: int = 260) -> None:
    documents = load_documents()
    # Явно создаем ноды для демонстрационного вывода, без embedding-модели и Qdrant.
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = node_parser.get_nodes_from_documents(documents)

    print(f"Документов: {len(documents)}")
    print(f"Нод после SentenceSplitter: {len(nodes)}")
    print(f"Показываю первые {min(limit, len(nodes))} нод:\n")

    for index, node in enumerate(nodes[:limit], start=1):
        meta = node.metadata
        text = node.get_content(metadata_mode="none").replace("\n", " ")
        print(f"--- NODE {index} ---")
        print(f"id: {node.node_id}")
        print(
            f"source: {meta.get('source')} | "
            f"year: {meta.get('year')} | category: {meta.get('category')}"
        )
        print(f"relationships: {format_relationships(node)}")
        print(f"chars: {len(text)}")
        print(text[:chars] + ("..." if len(text) > chars else ""))
        print()


def load_index(
    *,
    collection_name: str = COLLECTION_NAME,
    model: str = LLM_MODEL,
) -> VectorStoreIndex:
    configure_settings(model)
    client = qdrant_client()

    if not collection_exists(client, collection_name):
        raise RuntimeError(
            f"Коллекция {collection_name!r} не найдена. "
            "Сначала выполните: uv run python llamaindex/rag_llamaindex_demo.py index"
        )

    # Подключаемся к существующей Qdrant collection без повторной индексации.
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def metadata_filters(year: int | None = None, category: str | None = None) -> MetadataFilters | None:
    # MetadataFilters передаются retriever/query engine и транслируются в vector store.
    filters: list[MetadataFilter] = []
    if year is not None:
        filters.append(
            MetadataFilter(key="year", value=year, operator=FilterOperator.EQ)
        )
    if category:
        filters.append(
            MetadataFilter(key="category", value=category, operator=FilterOperator.EQ)
        )
    return MetadataFilters(filters=filters) if filters else None


def print_sources(source_nodes: Iterable[object]) -> None:
    print("\nИсточники:")
    for index, source_node in enumerate(source_nodes, start=1):
        # source_node - это NodeWithScore: найденная нода плюс similarity score.
        node = source_node.node
        meta = node.metadata
        score = getattr(source_node, "score", None)
        score_text = f"{score:.4f}" if isinstance(score, float) else "n/a"
        text = node.get_content(metadata_mode="none").replace("\n", " ")
        print(
            f"[{index}] score={score_text} "
            f"year={meta.get('year')} category={meta.get('category')} "
            f"source={meta.get('source')}"
        )
        print(f"    {text[:220]}...")


def retrieve(
    *,
    query: str,
    year: int | None = None,
    category: str | None = None,
    collection_name: str = COLLECTION_NAME,
    model: str = LLM_MODEL,
) -> None:
    index = load_index(collection_name=collection_name, model=model)
    # Retriever только ищет релевантные ноды, LLM здесь не вызывается.
    retriever = index.as_retriever(
        similarity_top_k=3,
        filters=metadata_filters(year=year, category=category),
    )
    nodes = retriever.retrieve(query)
    print_sources(nodes)


def query_rag(
    *,
    query: str,
    year: int | None = 2026,
    category: str | None = None,
    collection_name: str = COLLECTION_NAME,
    model: str = LLM_MODEL,
) -> None:
    index = load_index(collection_name=collection_name, model=model)
    # QueryEngine делает полный RAG: retrieval -> prompt -> LLM response.
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        filters=metadata_filters(year=year, category=category),
        text_qa_template=PromptTemplate(QA_TEMPLATE),
    )
    response = query_engine.query(query)

    print("\nОтвет:")
    print(str(response).strip())
    print_sources(response.source_nodes)


def run_demo(collection_name: str = COLLECTION_NAME, model: str = LLM_MODEL) -> None:
    client = qdrant_client()
    if not collection_exists(client, collection_name):
        print("Индекс еще не создан, запускаю первичную индексацию.")
        build_index(collection_name=collection_name, reset=False, model=model)

    question = "Сколько дней в неделю можно работать из дома и в какие дни?"

    print("\n=== Naive retrieval: без фильтра по году ===")
    retrieve(query="Какие правила удаленной работы?", collection_name=collection_name, model=model)

    print("\n=== Filtered retrieval: только year=2026 ===")
    retrieve(
        query="Какие правила удаленной работы?",
        year=2026,
        collection_name=collection_name,
        model=model,
    )

    print("\n=== RAG answer: LlamaIndex + локальная модель ===")
    query_rag(
        query=question,
        year=2026,
        collection_name=collection_name,
        model=model,
    )

    print("\n=== Negative test ===")
    query_rag(
        query="Какая зарплата у Senior Python Developer?",
        year=2026,
        collection_name=collection_name,
        model=model,
    )


def list_docs() -> None:
    documents = load_documents()
    print(f"Документов: {len(documents)}")
    for doc in documents:
        meta = doc.metadata
        print(
            f"- {meta['source']} | year={meta['year']} | "
            f"category={meta['category']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LlamaIndex RAG demo over data/knowledge_base."
    )
    parser.add_argument(
        "--model",
        default=LLM_MODEL,
        help=f"Локальная Ollama-модель для LlamaIndex LLM, default: {LLM_MODEL}",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Qdrant collection, default: {COLLECTION_NAME}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Проверить Ollama, модель и Qdrant.")
    subparsers.add_parser("list-docs", help="Показать текущие markdown-документы и metadata.")

    nodes_parser = subparsers.add_parser(
        "show-nodes",
        help="Показать LlamaIndex-ноды после разбиения документов на чанки.",
    )
    nodes_parser.add_argument("--limit", type=int, default=10)
    nodes_parser.add_argument("--chars", type=int, default=260)

    index_parser = subparsers.add_parser("index", help="Создать индекс LlamaIndex в Qdrant.")
    index_parser.add_argument(
        "--reset",
        action="store_true",
        help="Пересоздать коллекцию LlamaIndex в Qdrant, если она уже есть.",
    )

    retrieve_parser = subparsers.add_parser("retrieve", help="Показать найденные чанки без генерации.")
    retrieve_parser.add_argument("question")
    retrieve_parser.add_argument("--year", type=int)
    retrieve_parser.add_argument("--category")

    query_parser = subparsers.add_parser("query", help="Задать RAG-вопрос.")
    query_parser.add_argument("question")
    query_parser.add_argument("--year", type=int, default=2026)
    query_parser.add_argument(
        "--no-year-filter",
        action="store_true",
        help="Не ограничивать RAG-контекст годом. По умолчанию используется year=2026.",
    )
    query_parser.add_argument("--category")

    subparsers.add_parser("demo", help="Запустить полный сценарий: naive, filtered, RAG, negative test.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "check":
        return 0 if check_ollama(args.model) and check_qdrant() else 1
    if args.command == "list-docs":
        list_docs()
        return 0
    if args.command == "show-nodes":
        show_nodes(limit=args.limit, chars=args.chars)
        return 0
    if args.command == "index":
        build_index(collection_name=args.collection, reset=args.reset, model=args.model)
        return 0
    if args.command == "retrieve":
        retrieve(
            query=args.question,
            year=args.year,
            category=args.category,
            collection_name=args.collection,
            model=args.model,
        )
        return 0
    if args.command == "query":
        query_rag(
            query=args.question,
            year=None if args.no_year_filter else args.year,
            category=args.category,
            collection_name=args.collection,
            model=args.model,
        )
        return 0
    if args.command == "demo":
        run_demo(collection_name=args.collection, model=args.model)
        return 0

    raise AssertionError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
