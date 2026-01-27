import sys
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Конфигурация
COMPANY_NAME = "РобоТех"
TARGET_FILE_COUNT = 30
DATA_DIR = Path("data/knowledge_base")
MODEL_NAME = "mistral:7b"

# Инициализация LLM
print(f"⏳ Инициализация модели {MODEL_NAME}...")
try:
    llm = ChatOllama(model=MODEL_NAME, temperature=0.7)
except Exception as e:
    print(f"Ошибка инициализации LLM: {e}")
    sys.exit(1)


def _invoke_llm(system_prompt: str, user_prompt: str, title: str) -> str:
    """
    Универсальный вызов LLM с обработкой ошибок.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ],
    )

    try:
        response = (prompt | llm).invoke({})
        content = response.content
        return content if isinstance(content, str) else str(content)
    except Exception as e:
        print(f"Ошибка генерации для {title}: {e}")
        return f"# {title}\n\nОшибка генерации."


def generate_core_document(title: str, key_facts: str) -> str:
    """
    Генерирует основной документ с обязательными фактами.
    """
    system_prompt = (
        f"Ты — технический писатель в корпорации '{COMPANY_NAME}'. "
        "Твоя задача — написать внутренний документ на основе переданных фактов. "
        "Стиль: формальный, бюрократический, подробный. "
        "ВАЖНО: Ты обязан включить в текст ВСЕ перечисленные ключевые факты без искажений. "
        "Добавь вступление, заключение и форматирование Markdown."
    )

    user_prompt = f"Заголовок: {title}\nКлючевые факты:\n{key_facts}"
    return _invoke_llm(system_prompt, user_prompt, title)


def generate_filler_document(title: str) -> str:
    """
    Генерирует вспомогательный (шумовой) корпоративный документ.
    """
    system_prompt = (
        f"Ты — автор второстепенной корпоративной документации компании '{COMPANY_NAME}'. "
        "Документ не является нормативным и не содержит критичных фактов. "
        "Цель — создать правдоподобный внутренний текст (инструкцию или регламент). "
        "Стиль: корпоративный. Объем: 3–4 абзаца. "
        "Используй форматирование Markdown (списки, жирный текст)."
    )

    user_prompt = f"Тема документа: {title}"
    return _invoke_llm(system_prompt, user_prompt, title)


def get_core_documents_specs() -> list[dict[str, Any]]:
    return [
        {
            "filename": "HR_policy_2024.md",
            "title": "Политика удаленной работы (Архив 2024)",
            "facts": "Год документа: 2024. Статус: Архив. Правило: Разрешена полная удаленка (5 дней в неделю).",
        },
        {
            "filename": "HR_policy_2026.md",
            "title": "Политика гибридной работы (Действует с 2026)",
            "facts": "Год документа: 2026. Статус: Активен. Правило: Обязательно быть в офисе во Вторник, Среду, Четверг. Удаленка только Пн и Пт.",
        },
        {
            "filename": "Holiday_Schedule.md",
            "title": "График отпусков 2026",
            "facts": "Система подачи заявок: Jira-Holiday. Срок подачи: за 2 недели. Запрет: в декабре отпуска запрещены из-за релизов.",
        },
        {
            "filename": "Security_Passwords.md",
            "title": "Политика информационной безопасности: Пароли",
            "facts": "Мин. длина: 16 символов. Спецсимволы: обязательны. Срок действия: 90 дней. Повторное использование запрещено.",
        },
        {
            "filename": "Project_Red_Eye.md",
            "title": "Проект 'Красный Глаз' (Red Eye)",
            "facts": "Статус: Секретно. Суть: Зрение для дронов-доставщиков. Стек: PyTorch, CUDA. Ответственный: Виктор Цой.",
        },
        {
            "filename": "Meeting_Notes_Corp.md",
            "title": "Протокол встречи по Новогоднему корпоративу",
            "facts": "Дата: 10.12.2025. Решение: Деда Мороза заменит ИИ-аватар. Закупить 50 кг мандаринов.",
        },
    ]


def get_filler_titles() -> list[str]:
    return [
        "Инструкция по настройке IDE JetBrains",
        "Регламент использования кофемашины на кухне",
        "Памятка по оформлению больничного листа",
        "Инструкция по бронированию переговорок",
        "Список разрешенного программного обеспечения",
        "Правила дресс-кода (Friday Casual)",
        "Руководство по использованию VPN клиента",
        "Процесс онбординга новых сотрудников",
        "Политика чистого стола (Clean Desk Policy)",
        "Как заказать такси за счет компании",
        "Регламент код-ревью (Code Review)",
        "Инструкция по пожарной безопасности",
    ]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Директория данных: {DATA_DIR.resolve()}\n")

    specs = get_core_documents_specs()
    print(f"🚀 Начинаем генерацию CORE документов ({len(specs)} шт)...")

    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] Генерирую: {spec['title']}...", end=" ", flush=True)
        content = generate_core_document(spec["title"], spec["facts"])
        (DATA_DIR / spec["filename"]).write_text(content, encoding="utf-8")
        print("✅")

    fillers = get_filler_titles()
    needed_count = max(0, TARGET_FILE_COUNT - len(specs))

    print(f"\n🚀 Начинаем генерацию FILLER документов (нужно еще {needed_count} шт)...")

    for i in range(needed_count):
        title = fillers[i % len(fillers)]
        filename = f"Doc_Filler_{i + 1}.md"
        print(f"[{i + 1}/{needed_count}] Генерирую: {title}...", end=" ", flush=True)
        content = generate_filler_document(title)
        (DATA_DIR / filename).write_text(content, encoding="utf-8")
        print("✅")

    print(f"\n✨ Готово! Все файлы сохранены в {DATA_DIR}")


if __name__ == "__main__":
    main()
