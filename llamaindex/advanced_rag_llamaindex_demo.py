"""Advanced LlamaIndex RAG demo: filters, hybrid search, rerank, Neo4j graph."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from rag_llamaindex_demo import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    QA_TEMPLATE,
    QDRANT_URL,
    check_ollama,
    check_qdrant,
    configure_settings,
    load_documents,
    metadata_filters,
    print_sources,
    select_device,
)


HYBRID_COLLECTION = "robotex_docs_llamaindex_hybrid"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
SPARSE_MODEL = "Qdrant/bm25"
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


MANUAL_GRAPH_ENTITIES = [
    {
        "id": "company:robotex",
        "label": "Company",
        "name": "РобоТех",
        "kind": "Company",
        "content": "Вымышленная компания, для которой построена внутренняя база знаний RAG.",
        "aliases": ["РобоТех", "RoboTech"],
    },
    {
        "id": "person:maria_sokolova",
        "label": "Person",
        "name": "Мария Соколова",
        "kind": "Person",
        "content": "Владелец бизнес-требований Project Atlas и ответственная за HR-REMOTE-2026.",
        "aliases": ["Мария Соколова"],
    },
    {
        "id": "person:ivan_petrov",
        "label": "Person",
        "name": "Иван Петров",
        "kind": "Person",
        "content": "Технический лидер Project Atlas и ответственный за внедрение ZTA-17.",
        "aliases": ["Иван Петров"],
    },
    {
        "id": "person:anna_kim",
        "label": "Person",
        "name": "Анна Ким",
        "kind": "Person",
        "content": "Отвечает за бюджет Project Atlas и согласование нестандартных устройств.",
        "aliases": ["Анна Ким"],
    },
    {
        "id": "team:peopleops",
        "label": "Team",
        "name": "PeopleOps",
        "kind": "Team",
        "content": "Команда, отвечающая за HR-политики и правила гибридной работы.",
        "aliases": ["PeopleOps", "HR-департамент"],
    },
    {
        "id": "team:ai_platform",
        "label": "Team",
        "name": "AI Platform",
        "kind": "Team",
        "content": "Команда, отвечающая за RAG pipeline и проект Atlas.",
        "aliases": ["AI Platform"],
    },
    {
        "id": "team:security",
        "label": "Team",
        "name": "Security",
        "kind": "Team",
        "content": "Команда, отвечающая за требования Zero Trust Access и аудит доступа.",
        "aliases": ["Security", "команда Security"],
    },
    {
        "id": "team:platform",
        "label": "Team",
        "name": "Platform",
        "kind": "Team",
        "content": "Команда, где Иван Петров внедряет ZTA-17.",
        "aliases": ["Platform"],
    },
    {
        "id": "team:operations",
        "label": "Team",
        "name": "Operations",
        "kind": "Team",
        "content": "Владелец каталога устройств и закупочных процедур.",
        "aliases": ["Operations"],
    },
    {
        "id": "project:atlas",
        "label": "Project",
        "name": "Project Atlas",
        "kind": "Project",
        "content": "Graph RAG проект для связи документов, сотрудников, команд, политик и технических систем.",
        "aliases": ["Project Atlas", "проект Atlas", "Проект Atlas", "Atlas"],
    },
    {
        "id": "policy:hr_remote_2026",
        "label": "Policy",
        "name": "HR-REMOTE-2026",
        "kind": "Policy",
        "content": "Актуальная политика гибридной работы: удаленно по понедельникам и пятницам, офис во вторник-среду-четверг.",
        "aliases": ["HR-REMOTE-2026", "Политика гибридной и удаленной работы 2026"],
    },
    {
        "id": "policy:hr_remote_2024",
        "label": "Policy",
        "name": "HR-REMOTE-2024",
        "kind": "Policy",
        "content": "Архивная политика удаленной работы: один удаленный день в неделю, обычно пятница.",
        "aliases": ["HR-REMOTE-2024", "Архивная политика удаленной работы 2024"],
    },
    {
        "id": "code:wfh_mon_fri",
        "label": "Code",
        "name": "WFH-MON-FRI",
        "kind": "Code",
        "content": "Короткий код правила удаленной работы по понедельникам и пятницам.",
        "aliases": ["WFH-MON-FRI"],
    },
    {
        "id": "standard:zta_17",
        "label": "Standard",
        "name": "ZTA-17",
        "kind": "SecurityStandard",
        "content": "Стандарт Zero Trust Access 2026: SSO, FIDO2 и проверка устройства для инженерных сервисов.",
        "aliases": ["ZTA-17", "Zero Trust Access"],
    },
    {
        "id": "catalog:pr_4421_m2_16",
        "label": "CatalogItem",
        "name": "PR-4421-M2-16",
        "kind": "CatalogItem",
        "content": "Закупочная позиция Apple MacBook Air M2 16GB для ML/AI инженеров.",
        "aliases": ["PR-4421-M2-16", "Apple MacBook Air M2 16GB"],
    },
    {
        "id": "system:qdrant",
        "label": "System",
        "name": "Qdrant",
        "kind": "System",
        "content": "Vector DB для dense-векторов, sparse BM25 и hybrid search.",
        "aliases": ["Qdrant"],
    },
    {
        "id": "system:neo4j",
        "label": "System",
        "name": "Neo4j",
        "kind": "System",
        "content": "Graph database для property graph и визуализации связей.",
        "aliases": ["Neo4j"],
    },
    {
        "id": "system:ollama",
        "label": "System",
        "name": "Ollama",
        "kind": "System",
        "content": "Локальный runtime для LLM на машине пользователя.",
        "aliases": ["Ollama"],
    },
    {
        "id": "system:github_actions",
        "label": "System",
        "name": "GitHub Actions",
        "kind": "System",
        "content": "Инженерный сервис, доступ к которому проходит через ZTA-17.",
        "aliases": ["GitHub Actions"],
    },
    {
        "id": "tech:llamaindex_property_graph",
        "label": "Technology",
        "name": "LlamaIndex PropertyGraphIndex",
        "kind": "Technology",
        "content": "LlamaIndex индекс для работы с property graph.",
        "aliases": ["LlamaIndex PropertyGraphIndex", "PropertyGraphIndex"],
    },
    {
        "id": "tech:hybrid_search",
        "label": "Technology",
        "name": "Hybrid Search",
        "kind": "Technology",
        "content": "Поиск, объединяющий dense embeddings и sparse BM25 сигнал.",
        "aliases": ["hybrid search", "Hybrid Search"],
    },
    {
        "id": "tech:bm25",
        "label": "Technology",
        "name": "BM25",
        "kind": "Technology",
        "content": "Sparse keyword-сигнал для точных кодов вроде ZTA-17 и HR-REMOTE-2026.",
        "aliases": ["BM25"],
    },
    {
        "id": "tech:cross_encoder_reranking",
        "label": "Technology",
        "name": "Cross-Encoder Reranking",
        "kind": "Technology",
        "content": "Повторное ранжирование кандидатов после retrieval.",
        "aliases": ["Cross-Encoder Reranking", "reranking"],
    },
]


MANUAL_GRAPH_RELATIONS = [
    ("company:robotex", "HAS_TEAM", "team:peopleops", "PeopleOps ведет HR-политики."),
    ("company:robotex", "HAS_TEAM", "team:ai_platform", "AI Platform отвечает за RAG pipeline."),
    ("company:robotex", "HAS_TEAM", "team:security", "Security отвечает за ZTA-17."),
    ("company:robotex", "HAS_TEAM", "team:operations", "Operations отвечает за каталог устройств."),
    ("person:maria_sokolova", "OWNS", "policy:hr_remote_2026", "Мария Соколова ответственна за HR-REMOTE-2026."),
    ("team:peopleops", "OWNS", "policy:hr_remote_2026", "PeopleOps отвечает за политику гибридной работы."),
    ("policy:hr_remote_2026", "SUPERSEDES", "policy:hr_remote_2024", "Актуальная политика 2026 заменяет архив 2024."),
    ("policy:hr_remote_2026", "HAS_CODE", "code:wfh_mon_fri", "Короткий код политики: WFH-MON-FRI."),
    ("person:ivan_petrov", "LEADS", "project:atlas", "Иван Петров является техническим лидом Project Atlas."),
    ("person:maria_sokolova", "OWNS_REQUIREMENTS_FOR", "project:atlas", "Мария Соколова владеет бизнес-требованиями Atlas."),
    ("person:anna_kim", "MANAGES_BUDGET_FOR", "project:atlas", "Анна Ким отвечает за бюджет Atlas."),
    ("project:atlas", "USES", "system:neo4j", "Граф хранится в Neo4j."),
    ("project:atlas", "USES", "system:qdrant", "Векторные чанки и hybrid search хранятся в Qdrant."),
    ("project:atlas", "USES", "tech:llamaindex_property_graph", "Atlas использует LlamaIndex PropertyGraphIndex."),
    ("project:atlas", "FOLLOWS", "standard:zta_17", "Atlas следует требованиям ZTA-17."),
    ("team:ai_platform", "OWNS", "project:atlas", "AI Platform отвечает за RAG pipeline."),
    ("team:security", "OWNS", "standard:zta_17", "Security отвечает за требования ZTA-17."),
    ("person:ivan_petrov", "RESPONSIBLE_FOR", "standard:zta_17", "Иван Петров отвечает за внедрение ZTA-17."),
    ("standard:zta_17", "PROTECTS_ACCESS_TO", "system:qdrant", "Qdrant подключается через Zero Trust Access."),
    ("standard:zta_17", "PROTECTS_ACCESS_TO", "system:neo4j", "Neo4j подключается через Zero Trust Access."),
    ("standard:zta_17", "PROTECTS_ACCESS_TO", "system:github_actions", "GitHub Actions подключается через Zero Trust Access."),
    ("system:ollama", "RUNS_LOCAL_MODEL_FOR", "project:atlas", "Ollama обслуживает локальную LLM для RAG demo."),
    ("tech:hybrid_search", "USES", "tech:bm25", "Hybrid search использует sparse BM25-сигнал."),
    ("system:qdrant", "STORES", "tech:hybrid_search", "Qdrant хранит dense и sparse векторы для hybrid search."),
    ("tech:cross_encoder_reranking", "RERANKS_RESULTS_FROM", "tech:hybrid_search", "Cross-Encoder переставляет кандидатов после retrieval."),
    ("team:operations", "OWNS", "catalog:pr_4421_m2_16", "Operations владеет каталогом устройств."),
    ("person:anna_kim", "APPROVES", "catalog:pr_4421_m2_16", "Анна Ким согласует нестандартные устройства."),
]


def qdrant_clients() -> tuple[QdrantClient, AsyncQdrantClient]:
    client = QdrantClient(url=QDRANT_URL)
    aclient = AsyncQdrantClient(url=QDRANT_URL)
    return client, aclient


def hybrid_vector_store(collection_name: str = HYBRID_COLLECTION) -> QdrantVectorStore:
    client, aclient = qdrant_clients()
    # QdrantVectorStore хранит dense-вектор от BAAI/bge-m3 и sparse BM25-вектор.
    return QdrantVectorStore(
        collection_name=collection_name,
        client=client,
        aclient=aclient,
        enable_hybrid=True,
        fastembed_sparse_model=SPARSE_MODEL,
        batch_size=16,
    )


def collection_exists(collection_name: str) -> bool:
    client = QdrantClient(url=QDRANT_URL)
    return client.collection_exists(collection_name)


def build_hybrid_index(
    *,
    collection_name: str = HYBRID_COLLECTION,
    reset: bool = False,
    model: str = LLM_MODEL,
) -> VectorStoreIndex:
    configure_settings(model)
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(collection_name):
        if not reset:
            print(
                f"Коллекция {collection_name!r} уже существует. "
                "Для полной пересборки выполните index-hybrid --reset."
            )
            return load_hybrid_index(collection_name=collection_name, model=model)
        print(f"Удаляю hybrid-коллекцию {collection_name!r}...")
        client.delete_collection(collection_name)

    documents = load_documents()
    print(f"Документов для hybrid index: {len(documents)}")
    vector_store = hybrid_vector_store(collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print(f"Hybrid index готов: Qdrant collection={collection_name!r}")
    return index


def load_hybrid_index(
    *,
    collection_name: str = HYBRID_COLLECTION,
    model: str = LLM_MODEL,
) -> VectorStoreIndex:
    configure_settings(model)
    if not collection_exists(collection_name):
        raise RuntimeError(
            f"Коллекция {collection_name!r} не найдена. "
            "Сначала выполните: uv run python llamaindex/advanced_rag_llamaindex_demo.py index-hybrid"
        )
    return VectorStoreIndex.from_vector_store(vector_store=hybrid_vector_store(collection_name))


def print_nodes(title: str, nodes: Iterable[NodeWithScore]) -> None:
    print(f"\n=== {title} ===")
    print_sources(nodes)


def retrieve_nodes(
    *,
    query: str,
    index: VectorStoreIndex,
    top_k: int = 5,
    year: int | None = None,
    category: str | None = None,
    hybrid: bool = False,
) -> list[NodeWithScore]:
    retriever = index.as_retriever(
        similarity_top_k=top_k,
        sparse_top_k=max(8, top_k),
        hybrid_top_k=top_k,
        alpha=0.5,
        vector_store_query_mode=(
            VectorStoreQueryMode.HYBRID if hybrid else VectorStoreQueryMode.DEFAULT
        ),
        filters=metadata_filters(year=year, category=category),
    )
    return retriever.retrieve(query)


def compare_metadata_filters(
    query: str,
    *,
    collection_name: str = HYBRID_COLLECTION,
    model: str = LLM_MODEL,
) -> None:
    index = load_hybrid_index(collection_name=collection_name, model=model)
    print_nodes(
        "Без metadata filter: могут всплыть архивы и нерелевантные категории",
        retrieve_nodes(query=query, index=index, top_k=4, hybrid=True),
    )
    print_nodes(
        "Только архив year=2024: специально показываем неверный временной срез",
        retrieve_nodes(query=query, index=index, top_k=4, year=2024, category="HR", hybrid=True),
    )
    print_nodes(
        "Текущая HR-политика year=2026 + category=HR",
        retrieve_nodes(query=query, index=index, top_k=4, year=2026, category="HR", hybrid=True),
    )


def compare_hybrid_search(
    query: str,
    *,
    collection_name: str = HYBRID_COLLECTION,
    model: str = LLM_MODEL,
) -> None:
    index = load_hybrid_index(collection_name=collection_name, model=model)
    print_nodes(
        "Dense retrieval: только embedding similarity",
        retrieve_nodes(query=query, index=index, top_k=5, year=2026, hybrid=False),
    )
    print_nodes(
        "Hybrid retrieval: dense + sparse BM25 в Qdrant",
        retrieve_nodes(query=query, index=index, top_k=5, year=2026, hybrid=True),
    )


def rerank_search(
    query: str,
    *,
    collection_name: str = HYBRID_COLLECTION,
    model: str = LLM_MODEL,
    top_n: int = 3,
) -> list[NodeWithScore]:
    index = load_hybrid_index(collection_name=collection_name, model=model)
    candidates = retrieve_nodes(query=query, index=index, top_k=8, year=2026, hybrid=True)
    print_nodes("До Cross-Encoder Reranking: top-8 кандидатов из hybrid retrieval", candidates)

    # SentenceTransformerRerank вызывает cross-encoder и переставляет кандидатов.
    reranker = SentenceTransformerRerank(
        model=RERANK_MODEL,
        top_n=top_n,
        device=select_device(),
        keep_retrieval_score=True,
    )
    reranked = reranker.postprocess_nodes(candidates, query_str=query)
    print_nodes(f"После Cross-Encoder Reranking: top-{top_n}", reranked)
    return reranked


def query_hybrid_rerank(
    query: str,
    *,
    collection_name: str = HYBRID_COLLECTION,
    model: str = LLM_MODEL,
) -> None:
    index = load_hybrid_index(collection_name=collection_name, model=model)
    reranker = SentenceTransformerRerank(
        model=RERANK_MODEL,
        top_n=3,
        device=select_device(),
        keep_retrieval_score=True,
    )
    query_engine = index.as_query_engine(
        similarity_top_k=8,
        sparse_top_k=12,
        hybrid_top_k=8,
        alpha=0.5,
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        filters=metadata_filters(year=2026),
        node_postprocessors=[reranker],
        text_qa_template=PromptTemplate(QA_TEMPLATE),
    )
    response = query_engine.query(query)
    print("\nОтвет:")
    print(str(response).strip())
    print_sources(response.source_nodes)


def reset_neo4j_database() -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Не установлен neo4j driver. Выполните: uv sync") from exc

    driver = GraphDatabase.driver(
        NEO4J_URL,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()


def neo4j_driver():
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Не установлен neo4j driver. Выполните: uv sync") from exc

    return GraphDatabase.driver(
        NEO4J_URL,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )


def graph_label(label: str) -> str:
    allowed = {
        "CatalogItem",
        "Category",
        "Chunk",
        "Code",
        "Company",
        "Document",
        "Person",
        "Policy",
        "Project",
        "SecurityStandard",
        "Standard",
        "System",
        "Team",
        "Technology",
        "Year",
    }
    if label not in allowed:
        raise ValueError(f"Unsupported graph label: {label}")
    return label


def graph_relation_type(relation_type: str) -> str:
    if not relation_type.replace("_", "").isupper():
        raise ValueError(f"Unsupported relationship type: {relation_type}")
    return relation_type


def upsert_graph_node(session, *, node_id: str, label: str, properties: dict[str, object]) -> None:
    label = graph_label(label)
    session.run(
        f"""
        MERGE (n:`{label}` {{id: $id}})
        SET n += $properties,
            n.graph_node = true,
            n.visual_label = $label
        """,
        id=node_id,
        label=label,
        properties=properties,
    )


def upsert_graph_relation(
    session,
    *,
    start_id: str,
    relation_type: str,
    end_id: str,
    properties: dict[str, object] | None = None,
) -> None:
    relation_type = graph_relation_type(relation_type)
    session.run(
        f"""
        MATCH (a {{id: $start_id}})
        MATCH (b {{id: $end_id}})
        WHERE a.graph_node = true AND b.graph_node = true
        MERGE (a)-[r:`{relation_type}`]->(b)
        SET r += $properties
        """,
        start_id=start_id,
        end_id=end_id,
        properties=properties or {},
    )


def text_excerpt(text: str, limit: int = 700) -> str:
    clean = " ".join(text.split())
    return clean[:limit] + ("..." if len(clean) > limit else "")


def seed_manual_graph(reset: bool = False) -> None:
    if reset:
        print("Очищаю Neo4j database перед ручным seed graph...")
        reset_neo4j_database()

    documents = load_documents()
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = node_parser.get_nodes_from_documents(documents)

    driver = neo4j_driver()
    with driver.session() as session:
        for category in sorted({str(doc.metadata.get("category")) for doc in documents}):
            upsert_graph_node(
                session,
                node_id=f"category:{category}",
                label="Category",
                properties={
                    "name": category,
                    "kind": "Category",
                    "content": f"Metadata category={category}",
                },
            )

        for year in sorted({int(doc.metadata.get("year", 0)) for doc in documents}):
            upsert_graph_node(
                session,
                node_id=f"year:{year}",
                label="Year",
                properties={
                    "name": str(year),
                    "kind": "Year",
                    "content": f"Metadata year={year}",
                },
            )

        for doc in documents:
            source = str(doc.metadata["source"])
            content = doc.get_content(metadata_mode="none")
            title = content.splitlines()[0].lstrip("# ").strip() if content else source
            doc_id = f"doc:{source}"
            upsert_graph_node(
                session,
                node_id=doc_id,
                label="Document",
                properties={
                    "name": title,
                    "kind": "Document",
                    "source": source,
                    "filename": doc.metadata.get("filename"),
                    "year": doc.metadata.get("year"),
                    "category": doc.metadata.get("category"),
                    "doc_type": doc.metadata.get("doc_type", "document"),
                    "graph_ready": str(doc.metadata.get("graph_ready", "false")),
                    "content": text_excerpt(content, 900),
                },
            )
            upsert_graph_relation(
                session,
                start_id=doc_id,
                relation_type="IN_CATEGORY",
                end_id=f"category:{doc.metadata.get('category')}",
            )
            upsert_graph_relation(
                session,
                start_id=doc_id,
                relation_type="IN_YEAR",
                end_id=f"year:{doc.metadata.get('year')}",
            )

        previous_chunk_by_source: dict[str, str] = {}
        for index, chunk in enumerate(chunks, start=1):
            source = str(chunk.metadata["source"])
            chunk_id = f"chunk:{source}:{index}"
            content = chunk.get_content(metadata_mode="none")
            upsert_graph_node(
                session,
                node_id=chunk_id,
                label="Chunk",
                properties={
                    "name": f"{source} chunk {index}",
                    "kind": "LlamaIndex TextNode",
                    "source": source,
                    "year": chunk.metadata.get("year"),
                    "category": chunk.metadata.get("category"),
                    "chars": len(content),
                    "content": text_excerpt(content, 1100),
                },
            )
            upsert_graph_relation(
                session,
                start_id=f"doc:{source}",
                relation_type="HAS_CHUNK",
                end_id=chunk_id,
                properties={"order": index},
            )
            previous_chunk_id = previous_chunk_by_source.get(source)
            if previous_chunk_id:
                upsert_graph_relation(
                    session,
                    start_id=previous_chunk_id,
                    relation_type="NEXT",
                    end_id=chunk_id,
                )
            previous_chunk_by_source[source] = chunk_id

        for entity in MANUAL_GRAPH_ENTITIES:
            upsert_graph_node(
                session,
                node_id=str(entity["id"]),
                label=str(entity["label"]),
                properties={
                    "name": entity["name"],
                    "kind": entity["kind"],
                    "content": entity["content"],
                    "aliases": entity["aliases"],
                },
            )

        for start_id, relation_type, end_id, evidence in MANUAL_GRAPH_RELATIONS:
            upsert_graph_relation(
                session,
                start_id=start_id,
                relation_type=relation_type,
                end_id=end_id,
                properties={"evidence": evidence, "source": "manual_seed_from_documents"},
            )

        for index, chunk in enumerate(chunks, start=1):
            source = str(chunk.metadata["source"])
            chunk_id = f"chunk:{source}:{index}"
            content_lower = chunk.get_content(metadata_mode="none").lower()
            for entity in MANUAL_GRAPH_ENTITIES:
                aliases = [str(alias).lower() for alias in entity["aliases"]]
                if any(alias in content_lower for alias in aliases):
                    upsert_graph_relation(
                        session,
                        start_id=chunk_id,
                        relation_type="MENTIONS",
                        end_id=str(entity["id"]),
                        properties={
                            "source": source,
                            "evidence": f"Chunk text mentions {entity['name']}",
                        },
                    )

        stats = session.run(
            """
            MATCH (n)
            WHERE n.graph_node = true
            WITH count(n) AS nodes
            MATCH (a)-[r]->(b)
            WHERE a.graph_node = true AND b.graph_node = true
            RETURN nodes, count(r) AS relationships
            """
        ).single()

    driver.close()
    print(
        "Manual Neo4j graph готов: "
        f"nodes={stats['nodes']}, relationships={stats['relationships']}"
    )
    print("Откройте Neo4j Browser и выполните:")
    print("MATCH (n)-[r]->(m) WHERE n.graph_node = true AND m.graph_node = true RETURN n, r, m LIMIT 150")


def graph_stats() -> None:
    driver = neo4j_driver()
    with driver.session() as session:
        overview = session.run(
            """
            MATCH (n)
            WHERE n.graph_node = true
            WITH count(n) AS nodes
            MATCH (a)-[r]->(b)
            WHERE a.graph_node = true AND b.graph_node = true
            RETURN nodes, count(r) AS relationships
            """
        ).single()
        labels = session.run(
            """
            MATCH (n)
            WHERE n.graph_node = true
            UNWIND labels(n) AS label
            WITH label, count(*) AS count
            RETURN label, count
            ORDER BY count DESC, label
            """
        ).data()
        relations = session.run(
            """
            MATCH (a)-[r]->(b)
            WHERE a.graph_node = true AND b.graph_node = true
            RETURN type(r) AS type, count(*) AS count
            ORDER BY count DESC, type
            """
        ).data()

    driver.close()
    print(f"Neo4j graph: nodes={overview['nodes']}, relationships={overview['relationships']}")
    print("\nLabels:")
    for row in labels:
        print(f"- {row['label']}: {row['count']}")
    print("\nRelationships:")
    for row in relations:
        print(f"- {row['type']}: {row['count']}")


def graph_search(term: str) -> None:
    driver = neo4j_driver()
    with driver.session() as session:
        rows = session.run(
            """
            MATCH p=(n)-[r*1..2]-(m)
            WHERE n.graph_node = true
              AND m.graph_node = true
              AND (
                  toLower(coalesce(n.name, '') + ' ' + coalesce(n.content, ''))
                  CONTAINS toLower($term)
               OR toLower(coalesce(m.name, '') + ' ' + coalesce(m.content, ''))
                  CONTAINS toLower($term)
              )
            RETURN p
            LIMIT 25
            """,
            term=term,
        ).data()
    driver.close()

    print(f"Найдено путей: {len(rows)}")
    print("Для визуализации в Neo4j Browser выполните:")
    print(
        "MATCH p=(n)-[r*1..2]-(m) "
        "WHERE n.graph_node = true AND m.graph_node = true "
        f"AND (toLower(coalesce(n.name, '') + ' ' + coalesce(n.content, '')) CONTAINS toLower('{term}') "
        f"OR toLower(coalesce(m.name, '') + ' ' + coalesce(m.content, '')) CONTAINS toLower('{term}')) "
        "RETURN p LIMIT 25"
    )


def graph_check() -> bool:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("Neo4j driver не установлен. Выполните: uv sync")
        return False

    try:
        driver = GraphDatabase.driver(
            NEO4J_URL,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        with driver.session() as session:
            result = session.run("RETURN 1 AS ok").single()
        driver.close()
    except Exception as exc:
        print(f"Neo4j недоступен: {exc}")
        return False

    print(f"Neo4j OK: {NEO4J_URL}, result={result['ok']}")
    return True


def check_all(model: str = LLM_MODEL) -> int:
    ok = True
    ok = check_ollama(model) and ok
    ok = check_qdrant() and ok
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Hybrid sparse model: {SPARSE_MODEL}")
    try:
        import fastembed  # noqa: F401

        print("fastembed OK")
    except ImportError:
        print("fastembed не установлен. Выполните: uv sync")
        ok = False
    graph_check()
    return 0 if ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced LlamaIndex RAG demo.")
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--collection", default=HYBRID_COLLECTION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Проверить Ollama, Qdrant, fastembed и Neo4j.")

    index_parser = subparsers.add_parser("index-hybrid", help="Создать Qdrant hybrid index.")
    index_parser.add_argument("--reset", action="store_true")

    filters_parser = subparsers.add_parser("compare-filters", help="Показать эффект MetadataFilters.")
    filters_parser.add_argument("question")

    hybrid_parser = subparsers.add_parser("compare-hybrid", help="Сравнить dense и hybrid retrieval.")
    hybrid_parser.add_argument("question")

    rerank_parser = subparsers.add_parser("rerank", help="Показать порядок до и после reranking.")
    rerank_parser.add_argument("question")
    rerank_parser.add_argument("--top-n", type=int, default=3)

    query_parser = subparsers.add_parser("query-hybrid-rerank", help="RAG: hybrid retrieval + rerank + LLM.")
    query_parser.add_argument("question")

    graph_seed_parser = subparsers.add_parser("graph-seed", help="Быстро заполнить Neo4j ручным graph seed.")
    graph_seed_parser.add_argument("--reset", action="store_true")

    graph_search_parser = subparsers.add_parser("graph-search", help="Найти paths в ручном Neo4j graph.")
    graph_search_parser.add_argument("term")

    subparsers.add_parser("graph-stats", help="Показать количество graph nodes/relationships.")
    subparsers.add_parser("graph-check", help="Проверить Neo4j.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "check":
        return check_all(args.model)
    if args.command == "index-hybrid":
        build_hybrid_index(
            collection_name=args.collection,
            reset=args.reset,
            model=args.model,
        )
        return 0
    if args.command == "compare-filters":
        compare_metadata_filters(
            args.question,
            collection_name=args.collection,
            model=args.model,
        )
        return 0
    if args.command == "compare-hybrid":
        compare_hybrid_search(
            args.question,
            collection_name=args.collection,
            model=args.model,
        )
        return 0
    if args.command == "rerank":
        rerank_search(
            args.question,
            collection_name=args.collection,
            model=args.model,
            top_n=args.top_n,
        )
        return 0
    if args.command == "query-hybrid-rerank":
        query_hybrid_rerank(
            args.question,
            collection_name=args.collection,
            model=args.model,
        )
        return 0
    if args.command == "graph-seed":
        seed_manual_graph(reset=args.reset)
        return 0
    if args.command == "graph-search":
        graph_search(args.term)
        return 0
    if args.command == "graph-stats":
        graph_stats()
        return 0
    if args.command == "graph-check":
        return 0 if graph_check() else 1

    raise AssertionError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
