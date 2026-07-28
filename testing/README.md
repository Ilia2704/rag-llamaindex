# RAGAS, MLflow, Langfuse

Тесты используют текущую базу знаний `data/knowledge_base` и LlamaIndex retrieval из `llamaindex/rag_llamaindex_demo.py`.

## Локальный запуск

```bash
uv sync
docker compose up -d
uv run python -m testing.run_ragas_demo_test
```

По умолчанию:

- target model: Ollama `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M`;
- evaluator: YandexGPT Lite;
- embeddings for RAGAS: Yandex `text-embeddings/latest`;
- Qdrant collection: `robotex_docs_llamaindex`;
- MLflow tracking: `http://localhost:5001`;
- Langfuse: `http://localhost:3000`, если заполнены ключи `LANGFUSE_PUBLIC_KEY` и `LANGFUSE_SECRET_KEY`.

## Pre-deploy suite

```bash
uv run python -m testing.pre_deploy_test
```

Ограничить прогон одной моделью:

```bash
PREDEPLOY_MODEL_NAME=ollama-qwen3-4b-q4 uv run python -m testing.pre_deploy_test
```

Доступные имена матрицы:

- `ollama-qwen3-0.6b-q8`
- `ollama-qwen3-4b-q4`
- `yandexgpt-lite`

## Quality gates

Пороги задаются через `.env` или переменные CI:

```text
THRESH_FAITHFULNESS=0.80
THRESH_CONTEXT_RELEVANCE=0.50
THRESH_ANSWER_RELEVANCY=0.50
THRESH_CONTEXT_PRECISION=0.50
THRESH_CONTEXT_RECALL=0.70
THRESH_QA_SIM=0.80
```

## Reference-free режим

Можно запустить оценку без эталонных ответов:

```bash
RAGAS_REFERENCE_FREE=1 uv run python -m testing.run_ragas_demo_test
```

В этом режиме проверяются только метрики, которым не нужен `ground_truth`:

- `faithfulness`: ответ должен вытекать из найденного контекста;
- `context_relevance`: найденные чанки должны быть семантически близки к вопросу;
- `answer_relevancy`: ответ должен быть семантически близок к вопросу.

Метрики `context_recall` и `qa_semantic_correctness` отключаются, потому что они требуют эталонного ответа.
