"""Optional Langfuse wiring for LlamaIndex demos."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LangfuseConfig:
    public_key: str
    secret_key: str
    host: str
    trace_name: str


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def langfuse_config(trace_name: str) -> LangfuseConfig | None:
    public_key = _env("LANGFUSE_PUBLIC_KEY")
    secret_key = _env("LANGFUSE_SECRET_KEY")
    host = _env("LANGFUSE_HOST") or _env("LANGFUSE_BASE_URL", "http://localhost:3000")
    if not public_key or not secret_key:
        return None
    os.environ.setdefault("LANGFUSE_HOST", host)
    return LangfuseConfig(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        trace_name=trace_name,
    )


def configure_langfuse(trace_name: str) -> bool:
    """Attach Langfuse callback to LlamaIndex if credentials and integration exist."""
    config = langfuse_config(trace_name)
    if config is None:
        return False

    try:
        from llama_index.core import Settings
        from llama_index.core.callbacks import CallbackManager
    except Exception as exc:
        print(f"Langfuse tracing skipped: LlamaIndex callbacks unavailable ({exc})")
        return False

    handler = None
    import_errors: list[str] = []

    handler_candidates = [
        ("llama_index.callbacks.langfuse", "LangfuseCallbackHandler"),
        ("llama_index.core.callbacks.langfuse", "LangfuseCallbackHandler"),
        ("langfuse.llama_index", "LlamaIndexCallbackHandler"),
    ]
    for module_name, class_name in handler_candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            handler_cls = getattr(module, class_name)
        except Exception as exc:
            import_errors.append(f"{module_name}.{class_name}: {exc}")
            continue

        for kwargs in (
            {
                "public_key": config.public_key,
                "secret_key": config.secret_key,
                "host": config.host,
                "trace_name": config.trace_name,
            },
            {
                "public_key": config.public_key,
                "secret_key": config.secret_key,
                "host": config.host,
            },
            {},
        ):
            try:
                handler = handler_cls(**kwargs)
                break
            except TypeError:
                continue
        if handler is not None:
            break

    if handler is None:
        print("Langfuse tracing skipped: callback handler not found.")
        if import_errors:
            print("Checked integrations: " + " | ".join(import_errors[:2]))
        return False

    Settings.callback_manager = CallbackManager([handler])
    print(f"Langfuse tracing enabled: host={config.host}, trace={config.trace_name}")
    return True
