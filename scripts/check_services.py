import sys

import requests

MODEL_NAME = "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"


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

        # Проверка наличия локальной Ollama-модели
        print(f"⏳ Проверка модели {MODEL_NAME}...", end=" ")
        r = requests.get("http://localhost:11434/api/tags", timeout=30)
        models = [m["name"] for m in r.json()["models"]]

        # Ollama может вернуть точное имя или вариант с дополнительным суффиксом.
        # Ищем вхождение строки
        if any(MODEL_NAME in m for m in models):
            print("✅ Модель найдена")
            return True
        print(f"❌ Модель не найдена. Доступные: {models}")
        print(f"👉 Выполните: ollama pull {MODEL_NAME}")
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
