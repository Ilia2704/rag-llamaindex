# Workshop: Векторные БД и RAG

В этом воркшопе мы пройдем путь от создания векторного индекса до построения RAG-системы с фильтрацией метаданных и автоматической оценкой качества.

## Стек технологий

*   **LLM:** Qwen3 4B Q4_K_M (через Ollama)
*   **Vector DB:** Qdrant
*   **Embeddings:** BAAI/bge-m3
*   **Framework:** LangChain, LlamaIndex, uv
*   **Evaluation:** RAGAS, MLflow
*   **Observability:** Langfuse

## Быстрый старт

### 1. Предварительные требования
*   Установленный [Docker](https://www.docker.com/) и Docker Compose.
*   Установленная локально [Ollama](https://ollama.com/) (на macOS запускается как приложение/сервис на хосте).
*   Установленный [uv](https://github.com/astral-sh/uv) (современный менеджер пакетов Python).
*   Для Apple Silicon Ollama использует Metal-ускорение при локальном запуске на хосте.

### 2. Установка окружения

Клонируйте репозиторий и установите зависимости:

```bash
git clone https://github.com/pueraeternis/rag-engineering-workshop.git
cd rag-engineering-workshop

# Создание виртуального окружения и установка библиотек
uv sync
```

### 3. Запуск инфраструктуры

Ollama запускается локально на хосте и должна быть доступна на `http://localhost:11434`.
Скачайте модель Qwen3 4B Q4_K_M, если она еще не установлена:

```bash
ollama pull hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M
```

В Docker поднимаем Qdrant, MLflow и локальный Langfuse stack:

```bash
docker compose up -d
```

Версия Qdrant закреплена как `qdrant/qdrant:v1.16.2`, чтобы совпадать с `qdrant-client==1.16.2`.
MLflow доступен на `http://localhost:5001`, Langfuse - на `http://localhost:3000`.

Проверьте, что все сервисы доступны:
```bash
uv run scripts/check_services.py
```

### 4. Корпус документов

Мы будем работать с уже подготовленной базой знаний вымышленной корпорации "РобоТех".
Документы лежат в `data/knowledge_base` и входят в репозиторий.

### 5. Запуск воркшопа

Откройте ноутбук `notebooks/rag_workshop.ipynb` и следуйте инструкциям внутри.

### 6. Альтернативное демо на LlamaIndex

В папке `llamaindex/` есть отдельная реализация RAG поверх тех же документов, но через LlamaIndex:

```bash
uv run python llamaindex/rag_llamaindex_demo.py check
uv run python llamaindex/rag_llamaindex_demo.py index
uv run python llamaindex/rag_llamaindex_demo.py demo
```

Там же есть расширенное демо:

```bash
uv run python llamaindex/advanced_rag_llamaindex_demo.py index-hybrid
uv run python llamaindex/advanced_rag_llamaindex_demo.py compare-filters "Какие правила удаленной работы?"
uv run python llamaindex/advanced_rag_llamaindex_demo.py compare-hybrid "Что означает ZTA-17 и кто отвечает за внедрение?"
uv run python llamaindex/advanced_rag_llamaindex_demo.py rerank "Кто отвечает за проект Atlas и какие системы он использует?"
```

Подробная инструкция: `llamaindex/README.md`.

### 7. RAGAS/MLflow/Langfuse тестирование

Заполните `.env`:

```text
YC_API_KEY=<yandex cloud api key>
YC_FOLDER_ID=<yandex cloud folder id>
LANGFUSE_PUBLIC_KEY=<langfuse public key>
LANGFUSE_SECRET_KEY=<langfuse secret key>
```

Минимальный локальный запуск RAGAS quality gate:

```bash
uv sync
docker compose up -d
uv run python -m testing.run_ragas_demo_test
```

Pre-deploy suite с матрицей моделей и MLflow-логированием:

```bash
uv run python -m testing.pre_deploy_test
```

Ограничить pre-deploy одной моделью:

```bash
PREDEPLOY_MODEL_NAME=ollama-qwen3-4b-q4 uv run python -m testing.pre_deploy_test
```

CI/CD логика лежит в `.github/workflows/main.yml` и повторяет donor-подход:

- `model-tests`: self-hosted runner, матрица `ollama-qwen3-0.6b-q8`, `ollama-qwen3-4b-q4`, `yandexgpt-lite`;
- `testing.pre_deploy_test`: запускает `testing.run_ragas_demo_test`, пишет метрики/артефакты в MLflow;
- `testing.publish_model_placeholder`: после успешного прогона кладет placeholder модели в MinIO bucket Langfuse;
- `toxicity-test`: запускает `testing.toxic_test` и валит pipeline при `ГЕЙТ: ПРОВАЛ`.

Основные параметры quality gate:

```text
THRESH_FAITHFULNESS=0.80
THRESH_CONTEXT_RELEVANCE=0.50
THRESH_ANSWER_RELEVANCY=0.50
THRESH_CONTEXT_PRECISION=0.50
THRESH_CONTEXT_RECALL=0.70
THRESH_QA_SIM=0.80
```

RAGAS-кейсы используют текущую базу знаний `data/knowledge_base`, а не данные из архива. Если Qdrant collection `robotex_docs_llamaindex` еще не создана, тест соберет ее из текущих markdown-документов без изменения файлов в `data`.

Для оценки без эталонных ответов включите reference-free режим:

```bash
RAGAS_REFERENCE_FREE=1 uv run python -m testing.run_ragas_demo_test
```

Он проверяет `faithfulness`, `context_relevance` и `answer_relevancy`, а метрики, зависящие от `ground_truth`, пропускает.

## Сценарий практики

1.  **Ingestion:** Загрузка документов и извлечение метаданных (Год, Категория).
2.  **Indexing:** Настройка HNSW индекса в Qdrant вручную.
3.  **Naive Search:** Почему простой векторный поиск находит устаревшие документы?
4.  **Advanced Search:** Применение фильтров (`Metadata Filtering`) для отсечения неактуальной информации.
5.  **RAG Generation:** Генерация ответа с помощью Qwen3 4B.
6.  **Evaluation:** Использование паттерна "LLM-as-a-Judge" для оценки качества ответа.

---
Автор: Виталий Бабчук
