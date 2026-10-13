# Агентские демо: онбординг инженера Atlas

Общий сценарий: новый инженер выходит в команду Atlas. Нужно проверить доставку
ноутбука, найти ответственных за проект, используемые системы и требования доступа.
Заказы вымышленные: №271 — отправленный ноутбук, №272 — заказ в обработке.
Дата доставки фиксирована для учебного примера, а не является прогнозом.

## Два уровня демонстрации

| Модуль | Что показывает | Зависимости во время показа |
| --- | --- | --- |
| `toolcalling/` | Два инструмента оборудования, валидация, repair loop | Ollama |
| `multiagent/` | Планирование, три исполнителя, RAG, критик, адресное исправление | Ollama; Qdrant опционально |

Запуск из корня репозитория:

```bash
uv run python -m agentic.toolcalling.demo --inject-invalid-args
uv run python -m agentic.multiagent.demo --inject-invalid-args --inject-critic-error
```

Одна установленная модель `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M` обслуживает все роли.
Вызовы последовательные. Это разные агенты с отдельными инструкциями и контекстами,
а не несколько одновременно загруженных моделей.

## Структура

```text
agentic/
  common/trace.py           вертикальный trace и опциональный Langfuse
  toolcalling/demo.py      инструменты заказа и доставки, контракты, CLI
  retrieval/store.py       общий корпус РобоТех, local/Qdrant retrieval
  multiagent/schemas.py    контракты плана, подзадач, источников, критики и ответа
  multiagent/graph.py      LangGraph и самостоятельные циклы инструментов
  multiagent/demo.py       CLI полного онбординга
```

[Tool calling](toolcalling/README.md) · [Мультиагентный RAG](multiagent/README.md)

## Langfuse

По умолчанию Langfuse выключен: SDK и сеть не используются. Для подключения:

```bash
uv run python -m agentic.toolcalling.demo --inject-invalid-args --langfuse
uv run python -m agentic.multiagent.demo --inject-invalid-args --inject-critic-error --langfuse
```

В `.env` должны быть `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` и
`LANGFUSE_BASE_URL` либо `LANGFUSE_HOST` (по умолчанию `http://localhost:3000`).
Каждый запуск создаёт собственный корневой trace с именем:

- `agentic-atlas-toolcalling`;
- `agentic-atlas-multiagent`.

Внутри него события консольного trace сохраняются как дочерние observations,
LLM-вызовы передают Langfuse callback с входами, выходами и метаданными модели.
Мультиагентный запуск включает оборудование в свой общий trace, не создавая
отдельного корневого trace на каждого исполнителя. Роли и циклы указаны в событиях.

Перед инициализацией выполняется проверка health с таймаутом 1 секунда.
При отсутствии ключей или недоступности сервера выводится `LANGFUSE_SKIPPED`,
демо продолжает работу с консольным trace. Запускать Langfuse для показа не нужно.
Тесты проверяют интеграцию подставным SDK; доставка событий на реальный сервер
без запущенного Langfuse не проверялась.

## Проверки без моделей и сервисов

```bash
uv run python -m compileall -q agentic testing/test_agentic.py testing/test_toolcalling.py
uv run python -m unittest testing.test_agentic testing.test_toolcalling -v
```

Тесты проверяют реальные переходы LangGraph с подставными ответами модели,
настоящие локальные инструменты и корпус, успешный путь, неизвестный/неотправленный
заказ, ошибки аргументов, повторное выполнение, provenance цитат, права ролей,
пустой поиск, неправильный план, критика и её лимиты, сетевые ошибки, Langfuse
и работу без него. Адаптер Qdrant дополнительно проверяется на базе в памяти
с подставными embeddings. Это проверка кода оркестрации, не оценка качества локальной LLM.
Модель и embedding-модель при этих тестах не загружаются.
