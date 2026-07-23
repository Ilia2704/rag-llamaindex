# Workshop: Векторные БД и RAG

В этом воркшопе мы пройдем путь от создания векторного индекса до построения RAG-системы с фильтрацией метаданных и автоматической оценкой качества.

## Стек технологий

*   **LLM:** Qwen3 8B Q4_K_M (через Ollama)
*   **Vector DB:** Qdrant
*   **Embeddings:** BAAI/bge-m3
*   **Framework:** LangChain, LlamaIndex, uv

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
Скачайте модель Qwen3 8B Q4_K_M, если она еще не установлена:

```bash
ollama pull hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M
```

В Docker поднимаем только Qdrant:

```bash
docker compose up -d
```

Версия Qdrant закреплена как `qdrant/qdrant:v1.16.2`, чтобы совпадать с `qdrant-client==1.16.2`.

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

## Сценарий практики

1.  **Ingestion:** Загрузка документов и извлечение метаданных (Год, Категория).
2.  **Indexing:** Настройка HNSW индекса в Qdrant вручную.
3.  **Naive Search:** Почему простой векторный поиск находит устаревшие документы?
4.  **Advanced Search:** Применение фильтров (`Metadata Filtering`) для отсечения неактуальной информации.
5.  **RAG Generation:** Генерация ответа с помощью Qwen3 8B.
6.  **Evaluation:** Использование паттерна "LLM-as-a-Judge" для оценки качества ответа.

---
Автор: Виталий Бабчук
