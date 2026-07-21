# LlamaIndex RAG Demo

Эта папка добавляет второй вариант демо поверх тех же документов `data/knowledge_base/*.md`.
Исходный LangChain notebook не удаляется и не заменяется.

## Что реализовано

- загрузка текущих markdown-документов РобоТех;
- извлечение metadata `source`, `filename`, `year`, `category`;
- разбиение документов на чанки размером `1000` с overlap `200`;
- embedding через `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` в LlamaIndex;
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
ollama pull hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M
```

2. Поднимите Qdrant:

```bash
docker compose up -d
```

3. Установите зависимости:

```bash
uv sync
```

4. Если документов еще нет, сгенерируйте их текущим скриптом проекта:

```bash
uv run scripts/generate_data.py
```

## Проверка сервисов

```bash
uv run python llamaindex/rag_llamaindex_demo.py check
```

Проверяется:

- `http://localhost:11434/` отвечает;
- в Ollama есть `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M`;
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

По умолчанию используется `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M`. Можно передать другую локальную модель Ollama:

```bash
uv run python llamaindex/rag_llamaindex_demo.py --model llama3.1:8b query "Какие правила удаленной работы?"
```

Перед этим модель должна быть скачана:

```bash
ollama pull llama3.1:8b
```

## Как это сопоставляется с текущим LangChain notebook

- `data/knowledge_base/*.md` используются те же.
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` используется та же embedding-модель.
- `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M` используется та же локальная генеративная модель.
- Qdrant используется тот же локальный сервис.
- Коллекция другая: `robotex_docs_llamaindex`, чтобы не пересекаться с `robotex_docs` из notebook.
- Фильтр в LlamaIndex задается через `MetadataFilters`, а не через `qdrant_client.http.models.Filter`.
