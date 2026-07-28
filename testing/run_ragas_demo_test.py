"""
RAGAS quality gate for the current RoboTech RAG context.

The structure follows the donor project:
1. Generate answers with the target model.
2. Evaluate RAGAS metrics plus fallback answer relevancy.
3. Write optional JSON payload and MLflow metrics/artifacts.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness
from tqdm import tqdm

try:
    from langfuse.openai import OpenAI as LangfuseOpenAI
except Exception:
    LangfuseOpenAI = None


TESTING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTING_ROOT.parent
LLAMAINDEX_ROOT = REPO_ROOT / "llamaindex"
sys.path.insert(0, str(LLAMAINDEX_ROOT))

from rag_llamaindex_demo import (  # noqa: E402
    COLLECTION_NAME,
    LLM_MODEL,
    QA_TEMPLATE,
    build_index,
    collection_exists,
    load_index,
    metadata_filters,
    qdrant_client,
)


load_dotenv(REPO_ROOT / ".env")

warnings.filterwarnings(
    "ignore",
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"LangchainLLMWrapper is deprecated.*",
    category=DeprecationWarning,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ragas_robotex")

USE_SIMPLE_AR = (os.getenv("USE_SIMPLE_AR") or "1").strip().lower() not in {"0", "false"}
REFERENCE_FREE = (os.getenv("RAGAS_REFERENCE_FREE") or "0").strip().lower() in {"1", "true"}


@dataclass(frozen=True)
class Sample:
    question: str
    ground_truth: str = ""
    year: int | None = 2026
    category: str | None = None
    top_k: int = 4


SAMPLES: list[Sample] = [
    Sample(
        question="Сколько дней в неделю можно работать удаленно и в какие дни?",
        ground_truth="В 2026 году сотрудники инженерных, продуктовых и аналитических команд могут работать удаленно два дня в неделю: в понедельник и пятницу.",
        category="HR",
    ),
    Sample(
        question="Что означает ZTA-17 и что требуется для доступа к инженерным сервисам?",
        ground_truth="ZTA-17 - внутренний код стандарта Zero Trust Access 2026. Для доступа нужны корпоративный SSO, аппаратный ключ FIDO2 и активная проверка устройства.",
        category="IT_Security",
    ),
    Sample(
        question="Кто отвечает за проект Atlas и какие системы он использует?",
        ground_truth="Мария Соколова владеет бизнес-требованиями, Иван Петров является техническим лидом, Анна Ким отвечает за бюджет и закупки. Проект Atlas использует Neo4j и Qdrant.",
        category="Projects",
    ),
    Sample(
        question="Какой ноутбук является стандартным для ML-инженеров и какой код закупочной позиции?",
        ground_truth="Для ML-инженеров и AI-инженеров стандартный ноутбук - Apple MacBook Air M2 16GB. Код закупочной позиции: PR-4421-M2-16.",
        category="Procurement",
    ),
    Sample(
        question="Кто отвечает за политику HR-REMOTE-2026?",
        ground_truth="Владелец документа - департамент PeopleOps. Ответственная за политику HR-REMOTE-2026 - Мария Соколова.",
        category="HR",
    ),
]


def _prepare_langfuse_env() -> None:
    if not (os.getenv("LANGFUSE_HOST") or "").strip():
        host = (os.getenv("LANGFUSE_BASE_URL") or "").strip()
        if host:
            os.environ["LANGFUSE_HOST"] = host


def _langfuse_enabled() -> bool:
    return (
        LangfuseOpenAI is not None
        and bool((os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip())
        and bool((os.getenv("LANGFUSE_SECRET_KEY") or "").strip())
    )


def _get_openai_cls():
    _prepare_langfuse_env()
    if _langfuse_enabled():
        return LangfuseOpenAI
    return OpenAI


def _langfuse_request_args(*, name: str) -> dict[str, Any]:
    if not _langfuse_enabled():
        return {}
    return {"name": name}


def _normalize_provider(name: str, default: str) -> str:
    return (os.getenv(name, default) or default).strip().lower()


def _get_yandex_auth() -> tuple[str, str]:
    yc_api_key = (os.getenv("YC_API_KEY") or "").strip()
    yc_folder_id = (os.getenv("YC_FOLDER_ID") or "").strip()
    if not yc_api_key or not yc_folder_id:
        raise RuntimeError("Нужны YC_API_KEY и YC_FOLDER_ID в .env")
    return yc_api_key, yc_folder_id


def _default_chat_model(provider: str) -> str:
    if provider in {"yandex", "yandexgpt"}:
        _, folder_id = _get_yandex_auth()
        return f"gpt://{folder_id}/yandexgpt-lite/latest"
    if provider == "ollama":
        return LLM_MODEL
    raise RuntimeError(f"Неизвестный provider: {provider}")


def _default_embedding_model(provider: str) -> str:
    if provider in {"yandex", "yandexgpt"}:
        _, folder_id = _get_yandex_auth()
        return f"emb://{folder_id}/text-embeddings/latest"
    if provider == "ollama":
        return "nomic-embed-text"
    raise RuntimeError(f"Неизвестный embedding provider: {provider}")


def _make_openai_client(provider: str):
    client_cls = _get_openai_cls()
    if provider == "ollama":
        return client_cls(
            api_key=(os.getenv("OLLAMA_API_KEY") or "ollama").strip(),
            base_url=(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1").strip(),
        )
    if provider in {"yandex", "yandexgpt"}:
        yc_api_key, yc_folder_id = _get_yandex_auth()
        return client_cls(
            api_key="DUMMY",
            base_url="https://llm.api.cloud.yandex.net/v1",
            default_headers={
                "Authorization": f"Api-Key {yc_api_key}",
                "OpenAI-Project": yc_folder_id,
            },
        )
    raise RuntimeError(f"Неизвестный provider: {provider}")


def _evaluator_observation_name(eval_model: str) -> str:
    target_model = (os.getenv("OPENAI_MODEL") or "").strip() or "robotex-rag"
    return f"evaluator::{target_model}::via::{eval_model}"


def _make_chat_model(provider: str, model: str):
    if provider == "ollama":
        return ChatOpenAI(
            model=model,
            temperature=0,
            model_kwargs={"name": _evaluator_observation_name(model)},
            api_key=(os.getenv("OLLAMA_API_KEY") or "ollama").strip(),
            base_url=(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1").strip(),
        )
    if provider in {"yandex", "yandexgpt"}:
        yc_api_key, yc_folder_id = _get_yandex_auth()
        return ChatOpenAI(
            model=model,
            temperature=0,
            model_kwargs={"name": _evaluator_observation_name(model)},
            api_key="DUMMY",
            base_url="https://llm.api.cloud.yandex.net/v1",
            default_headers={
                "Authorization": f"Api-Key {yc_api_key}",
                "OpenAI-Project": yc_folder_id,
            },
        )
    raise RuntimeError(f"Неизвестный evaluator provider: {provider}")


def ensure_rag_index(collection_name: str, model: str):
    client = qdrant_client()
    if not collection_exists(client, collection_name):
        log.info("Qdrant collection %s not found; building from data/knowledge_base", collection_name)
        build_index(collection_name=collection_name, reset=False, model=model)
    return load_index(collection_name=collection_name, model=model)


def retrieve_contexts(index, sample: Sample) -> list[str]:
    retriever = index.as_retriever(
        similarity_top_k=sample.top_k,
        filters=metadata_filters(year=sample.year, category=sample.category),
    )
    nodes = retriever.retrieve(sample.question)
    return [node.node.get_content(metadata_mode="none") for node in nodes]


def llm_answer(
    question: str,
    contexts: list[str],
    *,
    client,
    model: str,
    case_id: int,
) -> str:
    joined_context = "\n\n---\n\n".join(contexts)
    prompt = QA_TEMPLATE.format(context_str=joined_context, query_str=question)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Ты - корпоративный ассистент РобоТех. Отвечай строго по контексту.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=500,
        **_langfuse_request_args(name=f"{model}::robotex_rag::case_{case_id:02d}"),
    )
    return (resp.choices[0].message.content or "").strip()


def cosine(u: list[float], v: list[float]) -> float:
    score = sum(a * b for a, b in zip(u, v))
    norm_u = math.sqrt(sum(a * a for a in u))
    norm_v = math.sqrt(sum(b * b for b in v))
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return max(0.0, min(1.0, (score / (norm_u * norm_v) + 1.0) / 2.0))


def embed_one(text: str, *, model: str, client, name: str) -> list[float]:
    result = client.embeddings.create(
        model=model,
        input=text,
        encoding_format="float",
        **_langfuse_request_args(name=name),
    )
    return result.data[0].embedding


def compute_simple_answer_relevancy_from_df(
    df_texts: pd.DataFrame,
    *,
    model: str,
    client,
    target_model: str,
) -> list[float]:
    scores: list[float] = []
    for idx, row in df_texts.iterrows():
        answer = str(row["answer"]).strip()
        if not answer:
            scores.append(0.0)
            continue
        q_vec = embed_one(
            str(row["question"]),
            model=model,
            client=client,
            name=f"{target_model}::embedding::question::case_{idx + 1:02d}",
        )
        a_vec = embed_one(
            answer,
            model=model,
            client=client,
            name=f"{target_model}::embedding::answer::case_{idx + 1:02d}",
        )
        scores.append(cosine(q_vec, a_vec))
    return scores


def compute_answer_gt_similarity(
    df_texts: pd.DataFrame,
    *,
    model: str,
    client,
    target_model: str,
) -> list[float]:
    scores: list[float] = []
    for idx, row in df_texts.iterrows():
        answer = str(row["answer"]).strip()
        if not answer:
            scores.append(0.0)
            continue
        a_vec = embed_one(
            answer,
            model=model,
            client=client,
            name=f"{target_model}::embedding::answer_gt::case_{idx + 1:02d}",
        )
        gt_vec = embed_one(
            str(row["ground_truth"]),
            model=model,
            client=client,
            name=f"{target_model}::embedding::ground_truth::case_{idx + 1:02d}",
        )
        scores.append(cosine(a_vec, gt_vec))
    return scores


def compute_context_relevance_from_df(
    df_texts: pd.DataFrame,
    *,
    model: str,
    client,
    target_model: str,
) -> list[float]:
    """Reference-free context relevance: mean cos_sim(question, retrieved_context_i)."""
    scores: list[float] = []
    for idx, row in df_texts.iterrows():
        question = str(row["question"]).strip()
        contexts = row["contexts"]
        if not question or not isinstance(contexts, list) or not contexts:
            scores.append(0.0)
            continue
        q_vec = embed_one(
            question,
            model=model,
            client=client,
            name=f"{target_model}::embedding::context_relevance_question::case_{idx + 1:02d}",
        )
        context_scores: list[float] = []
        for context_idx, context in enumerate(contexts, start=1):
            context_text = str(context).strip()
            if not context_text:
                continue
            c_vec = embed_one(
                context_text,
                model=model,
                client=client,
                name=(
                    f"{target_model}::embedding::context_relevance_context"
                    f"_{context_idx:02d}::case_{idx + 1:02d}"
                ),
            )
            context_scores.append(cosine(q_vec, c_vec))
        scores.append(sum(context_scores) / len(context_scores) if context_scores else 0.0)
    return scores


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _fmt(value: Any) -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "N/A"
        return f"{float(value):.4f}"
    except Exception:
        return "N/A"


def log_to_mlflow(payload: dict[str, Any], report: pd.DataFrame) -> None:
    enabled = (os.getenv("MLFLOW_ENABLED") or "1").strip().lower() not in {"0", "false"}
    if not enabled:
        return
    try:
        import mlflow
    except Exception as exc:
        log.warning("MLflow logging skipped: %s", exc)
        return

    try:
        tracking_uri = (os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5001").strip()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment((os.getenv("MLFLOW_EXPERIMENT_NAME") or "rag-engineering-workshop").strip())
        with tempfile.TemporaryDirectory(prefix="ragas_mlflow_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            payload_path = tmp_path / "ragas_payload.json"
            report_path = tmp_path / "ragas_report.csv"
            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            report.to_csv(report_path, index=False)
            with mlflow.start_run(run_name=(os.getenv("MLFLOW_RUN_NAME") or "ragas_robotex_test").strip()):
                mlflow.log_params(
                    {
                        "target_provider": payload.get("target_provider", ""),
                        "target_model": payload.get("target_model", ""),
                        "eval_provider": payload.get("eval_provider", ""),
                        "eval_model": payload.get("eval_model", ""),
                        "embedding_provider": payload.get("embedding_provider", ""),
                        "embedding_model": payload.get("embedding_model", ""),
                        "collection_name": payload.get("collection_name", ""),
                    }
                )
                mlflow.set_tags(
                    {
                        "test_type": "ragas_robotex_rag",
                        "status": "passed" if payload.get("passed") else "failed",
                        "deployment_ready": "true" if payload.get("passed") else "false",
                    }
                )
                mlflow.log_metric("test_passed", 1.0 if payload.get("passed") else 0.0)
                for metric_name, metric_value in payload.get("summary", {}).items():
                    mlflow.log_metric(metric_name, float(metric_value))
                mlflow.log_artifact(str(payload_path))
                mlflow.log_artifact(str(report_path))
    except Exception as exc:
        log.warning("MLflow logging skipped: %s", exc)


def main() -> None:
    target_provider = _normalize_provider("TARGET_PROVIDER", "ollama")
    eval_provider = _normalize_provider("EVAL_PROVIDER", "yandex")
    embedding_provider = _normalize_provider("EMBEDDING_PROVIDER", "yandex")

    target_model = (os.getenv("OPENAI_MODEL") or _default_chat_model(target_provider)).strip()
    eval_model = (os.getenv("EVAL_MODEL") or _default_chat_model(eval_provider)).strip()
    ragas_embedding_model = (
        os.getenv("RAGAS_EMBEDDING_MODEL") or _default_embedding_model(embedding_provider)
    ).strip()
    collection_name = (os.getenv("RAGAS_COLLECTION_NAME") or COLLECTION_NAME).strip()

    target_client = _make_openai_client(target_provider)
    embedding_client = _make_openai_client(embedding_provider)
    index = ensure_rag_index(collection_name, model=LLM_MODEL)

    rows: list[dict[str, Any]] = []
    print("\n[1/3] Генерация RAG-ответов по data/knowledge_base...")
    log.info(
        "cases=%d, collection=%s, target_provider=%s, target_model=%s, eval_provider=%s, eval_model=%s, embedding_provider=%s, embedding_model=%s",
        len(SAMPLES),
        collection_name,
        target_provider,
        target_model,
        eval_provider,
        eval_model,
        embedding_provider,
        ragas_embedding_model,
    )
    if REFERENCE_FREE:
        log.info("RAGAS reference-free mode enabled: ground_truth-dependent metrics are disabled")
    for case_id, sample in enumerate(tqdm(SAMPLES), start=1):
        contexts = retrieve_contexts(index, sample)
        answer = llm_answer(sample.question, contexts, client=target_client, model=target_model, case_id=case_id)
        rows.append(
            {
                "case_id": case_id,
                "question": sample.question,
                "answer": answer,
                "ground_truth": "" if REFERENCE_FREE else sample.ground_truth,
                "contexts": contexts,
            }
        )

    df = pd.DataFrame(rows)
    bad: list[tuple[int, str]] = []
    for index_id, row in df.iterrows():
        required_fields = ("question", "answer") if REFERENCE_FREE else ("question", "answer", "ground_truth")
        for field in required_fields:
            if not isinstance(row[field], str) or not row[field].strip():
                bad.append((index_id, field))
        if not isinstance(row["contexts"], list) or not all(isinstance(c, str) for c in row["contexts"]):
            bad.append((index_id, "contexts"))
    if bad:
        raise RuntimeError(f"Некорректные строки тестового набора: {bad}")

    print("\n[2/3] Оценка метрик...")
    evaluator_llm = LangchainLLMWrapper(_make_chat_model(eval_provider, eval_model))
    evaluator_embeddings = OpenAIEmbeddings(client=embedding_client, model=ragas_embedding_model)

    dataset_columns = ["question", "answer", "contexts"]
    if not REFERENCE_FREE:
        dataset_columns.append("ground_truth")
    hf_ds = Dataset.from_pandas(df[dataset_columns])

    metrics: list[Any] = [Faithfulness()]
    if not REFERENCE_FREE:
        metrics.extend([ContextPrecision(), ContextRecall()])
    if not USE_SIMPLE_AR:
        from ragas.metrics import AnswerRelevancy

        metrics.insert(1, AnswerRelevancy())

    result = evaluate(
        dataset=hf_ds,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=True,
        raise_exceptions=True,
        batch_size=4,
    )
    try:
        details_all = result.to_pandas()
    except AttributeError:
        details_all = pd.DataFrame(result)
    details_all.index = df.index

    details_all["answer_relevancy"] = compute_simple_answer_relevancy_from_df(
        df,
        model=ragas_embedding_model,
        client=embedding_client,
        target_model=target_model,
    )
    details_all["context_relevance"] = compute_context_relevance_from_df(
        df,
        model=ragas_embedding_model,
        client=embedding_client,
        target_model=target_model,
    )
    if not REFERENCE_FREE:
        details_all["qa_semantic_correctness"] = compute_answer_gt_similarity(
            df,
            model=ragas_embedding_model,
            client=embedding_client,
            target_model=target_model,
        )

    print("\n[3/3] Итоги (средние значения метрик):")
    wanted = [
        "faithfulness",
        "context_relevance",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "qa_semantic_correctness",
    ]
    present = [column for column in wanted if column in details_all.columns]
    summary = {column: float(details_all[column].mean(skipna=True)) for column in present}
    for metric_name in wanted:
        print(f"- {metric_name:23}: {_fmt(summary.get(metric_name))}")

    thresholds = {
        "faithfulness": env_float("THRESH_FAITHFULNESS", 0.80),
        "context_relevance": env_float("THRESH_CONTEXT_RELEVANCE", 0.50),
        "answer_relevancy": env_float("THRESH_ANSWER_RELEVANCY", 0.50),
        "context_precision": env_float("THRESH_CONTEXT_PRECISION", 0.50),
        "context_recall": env_float("THRESH_CONTEXT_RECALL", 0.70),
        "qa_semantic_correctness": env_float("THRESH_QA_SIM", 0.80),
    }

    report = df.join(details_all, how="left")
    print("\n=== Подробный отчёт по кейсам ===")
    case_results: list[dict[str, Any]] = []
    for index_id, row in report.iterrows():
        case_id = int(row.get("case_id", index_id + 1))
        print(f"\n[{case_id}] Q: {row['question']}")
        print(f"     A: {row['answer']}")
        if not REFERENCE_FREE:
            print(f"    GT: {row['ground_truth']}")
        metric_parts: list[str] = []
        metrics_for_case: dict[str, Any] = {}
        for metric_name in wanted:
            if metric_name not in report.columns:
                continue
            raw_value = row.get(metric_name)
            metric_parts.append(f"{metric_name}={_fmt(raw_value)}")
            try:
                if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
                    metrics_for_case[metric_name] = None
                else:
                    metrics_for_case[metric_name] = float(raw_value)
            except Exception:
                metrics_for_case[metric_name] = None
        print("Scores: " + (", ".join(metric_parts) if metric_parts else "нет доступных метрик"))
        case_results.append(
            {
                "case_id": case_id,
                "question": row["question"],
                "answer": row["answer"],
                "ground_truth": "" if REFERENCE_FREE else row["ground_truth"],
                "metrics": metrics_for_case,
            }
        )

    failed = [
        metric_name
        for metric_name, threshold in thresholds.items()
        if metric_name in present
        and (
            summary.get(metric_name) is None
            or math.isnan(float(summary[metric_name]))
            or float(summary[metric_name]) < threshold
        )
    ]

    result_payload = {
        "target_provider": target_provider,
        "target_model": target_model,
        "eval_provider": eval_provider,
        "eval_model": eval_model,
        "embedding_provider": embedding_provider,
        "embedding_model": ragas_embedding_model,
        "collection_name": collection_name,
        "reference_free": REFERENCE_FREE,
        "summary": summary,
        "cases": case_results,
        "thresholds": thresholds,
        "failed_metrics": failed,
        "passed": not failed,
    }

    result_json_path = (os.getenv("RESULT_JSON_PATH") or "").strip()
    if result_json_path:
        Path(result_json_path).write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    log_to_mlflow(result_payload, report)

    if failed:
        print("\n[FAIL] Порог(и) не пройдены:", failed)
        sys.exit(1)
    print("\n[OK] Все пороги пройдены.")


if __name__ == "__main__":
    main()
