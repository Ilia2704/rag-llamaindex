# 🧠 Workshop: Векторные БД и RAG

В этом воркшопе мы пройдем путь от создания векторного индекса до построения RAG-системы с фильтрацией метаданных и автоматической оценкой качества.

## 🛠 Стек технологий

*   **LLM:** Mistral 7B (через Ollama)
*   **Vector DB:** Qdrant
*   **Embeddings:** BAAI/bge-m3
*   **Framework:** LangChain, uv

## 🚀 Быстрый старт

### 1. Предварительные требования
*   Установленный [Docker](https://www.docker.com/) и Docker Compose.
*   Установленный [uv](https://github.com/astral-sh/uv) (современный менеджер пакетов Python).
*   Желательно: GPU (NVIDIA) для ускорения работы локальных моделей.

### 2. Установка окружения

Клонируйте репозиторий и установите зависимости:

```bash
git clone https://github.com/pueraeternis/rag-engineering-workshop.git
cd rag-engineering-workshop

# Создание виртуального окружения и установка библиотек
uv sync
```

### 3. Запуск инфраструктуры

Поднимаем Ollama и Qdrant в контейнерах:

```bash
docker compose up -d
```

> **Важно:** При первом запуске Ollama начнет скачивать модель Mistral 7B (~4GB).
> Проверить статус загрузки можно командой: `docker logs -f rag_ollama`

Скачайте модель, если она не подтянулась автоматически:
```bash
docker exec -it rag_ollama ollama pull mistral:7b
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

## 📚 Сценарий практики

1.  **Ingestion:** Загрузка документов и извлечение метаданных (Год, Категория).
2.  **Indexing:** Настройка HNSW индекса в Qdrant вручную.
3.  **Naive Search:** Почему простой векторный поиск находит устаревшие документы?
4.  **Advanced Search:** Применение фильтров (`Metadata Filtering`) для отсечения неактуальной информации.
5.  **RAG Generation:** Генерация ответа с помощью Mistral 7B.
6.  **Evaluation:** Использование паттерна "LLM-as-a-Judge" для оценки качества ответа.

---
Автор: Виталий Бабчук