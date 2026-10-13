from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"


class KnowledgeStore:
    """Local lexical retrieval by default; existing Qdrant index is optional."""
    def __init__(self, root=ROOT, backend="local"):
        self.root = Path(root)
        self.backend = backend
        self.documents = {}
        for path in sorted(self.root.glob("*.md")):
            if path.is_symlink():
                continue
            raw = path.read_text(encoding="utf-8")
            meta, body = {}, raw
            if raw.startswith("---\n"):
                parts = raw.split("---", 2)
                if len(parts) == 3:
                    body = parts[2].strip()
                    for line in parts[1].splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            meta[k.strip()] = v.strip()
            self.documents[path.name] = {"document_id": path.name, "text": body,
                                          "category": meta.get("category", "General"),
                                          "year": int(meta.get("year", 2026))}
        if not self.documents:
            raise ValueError("Корпус документов пуст")
        self.index = None

    def read(self, document_id, category):
        # Exact catalog lookup: arbitrary filesystem paths cannot be read.
        doc = self.documents.get(document_id)
        if doc is None or doc["category"] != category:
            raise ValueError("Документ отсутствует или недоступен этой роли")
        return dict(doc)

    def search(self, query, category, year=2026):
        if self.backend == "qdrant":
            return self._qdrant(query, category, year)
        terms = set(re.findall(r"[\w-]+", query.lower()))
        hits = []
        for doc in self.documents.values():
            if doc["category"] != category or doc["year"] != year:
                continue
            words = set(re.findall(r"[\w-]+", doc["text"].lower()))
            overlap = len(terms & words)
            if overlap:
                hits.append({**doc, "score": overlap / max(len(terms), 1)})
        return sorted(hits, key=lambda d: (-d["score"], d["document_id"]))[:3]

    def _qdrant(self, query, category, year):
        # Lazy initialization: local mode never loads embeddings or talks to Qdrant.
        from llama_index.core import VectorStoreIndex
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        if self.index is None:
            import os
            client = QdrantClient(url=os.getenv("AGENTIC_QDRANT_URL", "http://localhost:6333"))
            collection = os.getenv("AGENTIC_COLLECTION", "robotex_docs_llamaindex")
            if not client.collection_exists(collection):
                raise ValueError("Нет RAG-индекса: сначала запустите существующее llamaindex index демо")
            embedding = HuggingFaceEmbedding(model_name="BAAI/bge-m3", normalize=True)
            self.index = VectorStoreIndex.from_vector_store(
                QdrantVectorStore(client=client, collection_name=collection), embed_model=embedding)
        filters = MetadataFilters(filters=[MetadataFilter(key="category", value=category),
                                          MetadataFilter(key="year", value=year)])
        nodes = self.index.as_retriever(similarity_top_k=3, filters=filters).retrieve(query)
        hits = []
        for item in nodes:
            ident = item.node.metadata.get("filename")
            if ident in self.documents:
                doc = self.read(ident, category)
                if doc["year"] == year:
                    hits.append({**doc, "text": item.node.get_content(metadata_mode="none"),
                                 "score": float(item.score or 0)})
        return hits
