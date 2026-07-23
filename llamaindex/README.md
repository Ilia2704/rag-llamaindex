# LlamaIndex RAG Demo

Эта папка добавляет второй вариант демо поверх тех же документов `data/knowledge_base/*.md`.
Исходный LangChain notebook не удаляется и не заменяется.

## Что реализовано

- загрузка текущих markdown-документов РобоТех;
- извлечение metadata `source`, `filename`, `year`, `category`;
- разбиение документов на чанки размером `1000` с overlap `200`;
- embedding через `BAAI/bge-m3` в LlamaIndex;
- хранение векторов в отдельной Qdrant-коллекции `robotex_docs_llamaindex`;
- локальная LLM через `llama_index.llms.ollama.Ollama`;
- naive retrieval без фильтра;
- filtered retrieval с `year=2026`;
- RAG-ответ с промптом, который запрещает придумывать факты;
- negative test на вопрос, которого нет в документах.

Важно: модель по-прежнему физически обслуживает локальная Ollama на `localhost:11434`, но код обращается к ней через LlamaIndex-интеграцию, а не через LangChain.

## Подготовка

Из корня проекта:

```bash
cd /Users/newuser/Documents/repo/rag-engineering-workshop
```

1. Проверьте, что Ollama запущена и модель скачана:

```bash
ollama pull hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M
```

2. Поднимите Qdrant:

```bash
docker compose up -d
```

3. Установите зависимости:

```bash
uv sync
```

4. Проверьте, что корпус документов есть в `data/knowledge_base`.

## Проверка сервисов

```bash
uv run python llamaindex/rag_llamaindex_demo.py check
```

Проверяется:

- `http://localhost:11434/` отвечает;
- в Ollama есть `hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M`;
- `http://localhost:6333/collections` отвечает.

## Проверка текущих документов

```bash
uv run python llamaindex/rag_llamaindex_demo.py list-docs
```

Команда показывает все markdown-файлы и metadata, которые попадут в индекс.

## Просмотр LlamaIndex-нод

```bash
uv run python llamaindex/rag_llamaindex_demo.py show-nodes
```

Команда показывает, как `Document` превращаются в `Node` после `SentenceSplitter`.
Это полезно для объяснения ingestion-этапа до записи в Qdrant.
Embedding-модель на этом шаге не загружается.
В выводе также показываются типы отношений ноды, например `SOURCE`, `PREVIOUS`, `NEXT`, если LlamaIndex создал эти связи.

Ограничить количество нод и длину текста:

```bash
uv run python llamaindex/rag_llamaindex_demo.py show-nodes --limit 5 --chars 180
```

## Индексация

```bash
uv run python llamaindex/rag_llamaindex_demo.py index
```

Создается отдельная коллекция:

```text
robotex_docs_llamaindex
```

Если коллекция уже есть, команда не добавляет документы повторно, чтобы не создать дубликаты.
Для полной пересборки индекса:

```bash
uv run python llamaindex/rag_llamaindex_demo.py index --reset
```

`--reset` удаляет только эту отдельную Qdrant-коллекцию, файлы проекта и документы не трогает.

## Retrieval без генерации

Naive retrieval:

```bash
uv run python llamaindex/rag_llamaindex_demo.py retrieve "Какие правила удаленной работы?"
```

Filtered retrieval только по актуальным документам 2026 года:

```bash
uv run python llamaindex/rag_llamaindex_demo.py retrieve "Какие правила удаленной работы?" --year 2026
```

Фильтр по категории:

```bash
uv run python llamaindex/rag_llamaindex_demo.py retrieve "Какие правила удаленной работы?" --year 2026 --category HR
```

## RAG-вопрос

```bash
uv run python llamaindex/rag_llamaindex_demo.py query "Сколько дней в неделю можно работать из дома и в какие дни?"
```

По умолчанию `query` использует фильтр `year=2026`, чтобы ответ строился по актуальной политике.

Вопрос без фильтра по году:

```bash
uv run python llamaindex/rag_llamaindex_demo.py query "Какие правила удаленной работы?" --no-year-filter
```

Команда `query` по умолчанию демонстрирует production-паттерн с актуальным срезом знаний, поэтому использует `year=2026`.

## Полный сценарий демо

```bash
uv run python llamaindex/rag_llamaindex_demo.py demo
```

Сценарий делает:

1. проверяет, есть ли индекс;
2. если индекса нет, создает его;
3. показывает naive retrieval без фильтра;
4. показывает filtered retrieval с `year=2026`;
5. генерирует RAG-ответ через локальную модель;
6. запускает negative test на вопрос про зарплату, которой нет в документах.

## Замена модели

По умолчанию используется `hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M`. Можно передать другую локальную модель Ollama:

```bash
uv run python llamaindex/rag_llamaindex_demo.py --model llama3.1:8b query "Какие правила удаленной работы?"
```

Перед этим модель должна быть скачана:

```bash
ollama pull llama3.1:8b
```

## Как это сопоставляется с текущим LangChain notebook

- `data/knowledge_base/*.md` используются те же.
- `BAAI/bge-m3` используется та же embedding-модель.
- `hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M` используется та же локальная генеративная модель.
- Qdrant используется тот же локальный сервис.
- Коллекция другая: `robotex_docs_llamaindex`, чтобы не пересекаться с `robotex_docs` из notebook.
- Фильтр в LlamaIndex задается через `MetadataFilters`, а не через `qdrant_client.http.models.Filter`.

## Расширенное демо: Filters + Hybrid + Rerank + Graph

Файл `advanced_rag_llamaindex_demo.py` развивает базовый пример и показывает четыре production-приема:

- `MetadataFilters`: ограничение поиска годом, категорией и другими metadata;
- `Hybrid Search`: dense-векторы `BAAI/bge-m3` + sparse BM25-сигнал в Qdrant;
- `Cross-Encoder Reranking`: повторное ранжирование найденных кандидатов;
- `Graph Retrieval`: ручной property graph в Neo4j поверх тех же документов.

Для наглядности в `data/knowledge_base/` лежат документы из нескольких доменов: HR, IT Security, Procurement, Projects и общие корпоративные инструкции.

### Важное про пересборку индексов

Embedding-модель снова `BAAI/bge-m3`, размерность вектора `1024`. Если раньше индекс строился на MiniLM с размерностью `384`, коллекции нужно пересоздать:

```bash
uv run python llamaindex/rag_llamaindex_demo.py index --reset
uv run python llamaindex/advanced_rag_llamaindex_demo.py index-hybrid --reset
```

### Подготовка расширенного демо

```bash
ollama pull hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M
uv sync
docker compose up -d
```

Qdrant закреплен на `qdrant/qdrant:v1.16.2`, потому что Python-клиент закреплен как `qdrant-client==1.16.2`.
Если раньше был поднят `qdrant/qdrant:latest`, пересоздайте контейнер и volume:

```bash
docker compose down -v
docker compose up -d
```

Проверка:

```bash
uv run python llamaindex/advanced_rag_llamaindex_demo.py check
```

### Metadata filtering

```bash
uv run python llamaindex/advanced_rag_llamaindex_demo.py compare-filters "Какие правила удаленной работы?"
```

Команда выводит три блока:

1. поиск без фильтра;
2. намеренно неверный архивный срез `year=2024`;
3. актуальный срез `year=2026 + category=HR`.

Так видно, почему фильтр нужен не для красоты, а для отсечения устаревших правил.

### Hybrid Search

Сначала создайте hybrid-коллекцию:

```bash
uv run python llamaindex/advanced_rag_llamaindex_demo.py index-hybrid
```

Затем сравните dense и hybrid retrieval:

```bash
uv run python llamaindex/advanced_rag_llamaindex_demo.py compare-hybrid "Что означает ZTA-17 и кто отвечает за внедрение?"
```

Dense retrieval ищет по смысловой близости. Hybrid retrieval добавляет sparse BM25-сигнал, поэтому лучше цепляется за точные коды вроде `ZTA-17`, `HR-REMOTE-2026`, `PR-4421-M2-16`.

### Cross-Encoder Reranking

```bash
uv run python llamaindex/advanced_rag_llamaindex_demo.py rerank "Кто отвечает за проект Atlas и какие системы он использует?"
```

Команда сначала показывает top-8 кандидатов из hybrid retrieval, затем показывает top-3 после cross-encoder reranking.

Полный RAG с hybrid retrieval и rerank:

```bash
uv run python llamaindex/advanced_rag_llamaindex_demo.py query-hybrid-rerank "Кто отвечает за проект Atlas и какие системы он использует?"
```

### Graph Retrieval: Neo4j manual property graph

Neo4j добавлен в `docker-compose.yaml` как optional profile, поэтому обычный Qdrant-запуск не меняется.
Образ закреплен как `neo4j:5.26.28-community`. Точная фиксация здесь нужна для воспроизводимости Docker-окружения; предупреждение о несовместимости относилось к Qdrant, поэтому Qdrant закреплен отдельно как `qdrant/qdrant:v1.16.2` под `qdrant-client==1.16.2`.

Запуск Neo4j:

```bash
docker compose --profile graph up -d neo4j
```

Проверка Neo4j:

```bash
uv run python llamaindex/advanced_rag_llamaindex_demo.py graph-check
```

Быстрое заполнение Neo4j ручным graph seed из текущих документов:

```bash
uv run python llamaindex/advanced_rag_llamaindex_demo.py graph-seed --reset
```

Эта команда создает:

- `Document` ноды для всех markdown-файлов;
- `Chunk` ноды, похожие на LlamaIndex `TextNode`;
- `Person`, `Team`, `Project`, `Policy`, `System`, `Technology` и другие entity-ноды;
- связи `HAS_CHUNK`, `MENTIONS`, `OWNS`, `LEADS`, `USES`, `FOLLOWS`, `SUPERSEDES`.

Граф заполняется вручную через официальный Python driver `neo4j`, а не через LlamaIndex Neo4j connector. Это сделано специально для демо: LLM-based extraction через `PropertyGraphIndex` на локальной Qwen3 8B оказался слишком долгим и нестабильным по памяти на MacBook Air M2. Ручной seed делает граф воспроизводимым: одни и те же документы дают одни и те же ноды, связи и цвета в Neo4j Browser.

Проверить статистику:

```bash
uv run python llamaindex/advanced_rag_llamaindex_demo.py graph-stats
```

Найти paths по свойствам нод:

```bash
uv run python llamaindex/advanced_rag_llamaindex_demo.py graph-search Atlas
```

В Neo4j Browser удобно начать с:

```cypher
MATCH (n)-[r]->(m)
WHERE n.graph_node = true AND m.graph_node = true
RETURN n, r, m
LIMIT 150
```

Для Atlas:

```cypher
MATCH p=(n)-[r*1..2]-(m)
WHERE n.graph_node = true
  AND m.graph_node = true
  AND (
    toLower(coalesce(n.name, '') + ' ' + coalesce(n.content, '')) CONTAINS 'atlas'
    OR toLower(coalesce(m.name, '') + ' ' + coalesce(m.content, '')) CONTAINS 'atlas'
  )
RETURN p
LIMIT 25
```

Neo4j в этом демо помогает показать graph context: какие документы распались на чанки, какие сущности упоминаются в чанках и как сущности связаны между собой. Для генерации ответа используется отдельный RAG-пайплайн `query-hybrid-rerank`.
