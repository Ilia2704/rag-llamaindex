# Workshop: Векторные БД и RAG

В этом воркшопе мы пройдем путь от создания векторного индекса до построения RAG-системы с фильтрацией метаданных и автоматической оценкой качества.

## Стек технологий

*   **LLM:** Qwen3 4B Q4_K_M (через Ollama)
*   **Vector DB:** Qdrant
*   **Embeddings:** sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
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
Скачайте модель Qwen3 4B Q4_K_M, если она еще не установлена:

```bash
ollama pull hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M
```

В Docker поднимаем только Qdrant:

```bash
docker compose up -d
```

Проверьте, что все сервисы доступны:
```bash
uv run scripts/check_services.py
```

### 4. Генерация данных

Мы будем работать с документацией вымышленной корпорации "РобоТех". Сгенерируйте датасет с помощью LLM:

```bash
uv run scripts/generate_data.py
```
*Это создаст 30 markdown-файлов в папке `data/knowledge_base`.*

### 5. Запуск воркшопа

Откройте ноутбук `notebooks/rag_workshop.ipynb` и следуйте инструкциям внутри.

### 6. Альтернативное демо на LlamaIndex

В папке `llamaindex/` есть отдельная реализация RAG поверх тех же документов, но через LlamaIndex:

```bash
uv run python llamaindex/rag_llamaindex_demo.py check
uv run python llamaindex/rag_llamaindex_demo.py index
uv run python llamaindex/rag_llamaindex_demo.py demo
```

Подробная инструкция: `llamaindex/README.md`.

## Сценарий практики

1.  **Ingestion:** Загрузка документов и извлечение метаданных (Год, Категория).
2.  **Indexing:** Настройка HNSW индекса в Qdrant вручную.
3.  **Naive Search:** Почему простой векторный поиск находит устаревшие документы?
4.  **Advanced Search:** Применение фильтров (`Metadata Filtering`) для отсечения неактуальной информации.
5.  **RAG Generation:** Генерация ответа с помощью Qwen3 4B.
6.  **Evaluation:** Использование паттерна "LLM-as-a-Judge" для оценки качества ответа.

---
Автор: Виталий Бабчук
