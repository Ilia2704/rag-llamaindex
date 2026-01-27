import sys

import requests


def check_ollama():
    print("⏳ Проверка Ollama...", end=" ")
    try:
        # Проверка доступности API
        r = requests.get("http://localhost:11434/", timeout=30)
        if r.status_code == 200:  # noqa: PLR2004
            print("✅ OK")
        else:
            print(f"❌ Ошибка: статус {r.status_code}")
            return False

        # Проверка наличия модели mistral:7b
        print("⏳ Проверка модели mistral:7b...", end=" ")
        r = requests.get("http://localhost:11434/api/tags", timeout=30)
        models = [m["name"] for m in r.json()["models"]]

        # Ollama может вернуть 'mistral:7b' или 'mistral:7b-instruct' и т.д.
        # Ищем вхождение строки
        if any("mistral:7b" in m for m in models):
            print("✅ Модель найдена")
            return True
        print(f"❌ Модель не найдена. Доступные: {models}")
        print("👉 Выполните: docker exec -it rag_ollama ollama pull mistral:7b")
        return False

    except Exception as e:
        print(f"❌ Ошибка соединения: {e}")
        return False


def check_qdrant():
    print("⏳ Проверка Qdrant...", end=" ")
    try:
        r = requests.get("http://localhost:6333/collections", timeout=30)
        if r.status_code == 200:  # noqa: PLR2004
            print(f"✅ OK (Коллекций: {len(r.json()['result']['collections'])})")
            return True
        print(f"❌ Ошибка: статус {r.status_code}")
        return False
    except Exception as e:
        print(f"❌ Ошибка соединения: {e}")
        return False


if __name__ == "__main__":
    ollama_ok = check_ollama()
    qdrant_ok = check_qdrant()

    if ollama_ok and qdrant_ok:
        print("\n🚀 Все системы готовы к работе!")
    else:
        print("\n⚠️ Есть проблемы с сервисами. Исправьте их перед продолжением.")
        sys.exit(1)
