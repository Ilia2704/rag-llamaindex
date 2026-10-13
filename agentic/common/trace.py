"""Vertical console events and opt-in Langfuse observations."""
from contextlib import contextmanager, nullcontext, ExitStack
import json
import os
import sys
from urllib.request import urlopen


class Trace:
    def __init__(self):
        self.index = 0
        self.client = None
        self.config = None

    def emit(self, event, data=None):
        self.index += 1
        print(f"[{self.index:02d}] {event}", file=sys.stderr, flush=True)
        if data is not None:
            for line in json.dumps(data, ensure_ascii=False, indent=2, default=str).splitlines():
                print(f"     {line}", file=sys.stderr, flush=True)
        if self.client:
            try:
                with self.client.start_as_current_observation(name=event, input=data):
                    pass
            except Exception:
                self.client = None
                self.config = None

    def invoke(self, runnable, messages):
        if self.config:
            return runnable.invoke(messages, config=self.config)
        return runnable.invoke(messages)

    @contextmanager
    def session(self, name, question, enabled=False):
        """No network or SDK initialization unless explicitly requested."""
        context = nullcontext()
        if enabled:
            try:
                if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
                    raise ValueError("Не заданы ключи Langfuse")
                host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST", "http://localhost:3000")
                with urlopen(host.rstrip("/") + "/api/public/health", timeout=1) as response:
                    if response.status != 200:
                        raise ValueError("Langfuse недоступен")
                from langfuse import Langfuse
                from langfuse.langchain import CallbackHandler
                self.client = Langfuse(public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                                       secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                                       base_url=host, timeout=2)
                self.config = {"callbacks": [CallbackHandler(public_key=os.environ["LANGFUSE_PUBLIC_KEY"])],
                               "run_name": name}
                context = self.client.start_as_current_observation(name=name, input=question)
            except Exception as exc:
                self.client = None
                self.config = None
                self.emit("LANGFUSE_SKIPPED", str(exc))
        else:
            self.emit("LANGFUSE_DISABLED", "Для подключения используйте --langfuse")
        try:
            with ExitStack() as stack:
                try:
                    root = stack.enter_context(context)
                    if root:
                        root.update_trace(name=name, tags=["agentic", name])
                except Exception as exc:
                    self.client = None
                    self.config = None
                    self.emit("LANGFUSE_SKIPPED", str(exc))
                yield self
        finally:
            if self.client:
                try:
                    self.client.flush()
                except Exception:
                    pass
            self.config = None
            self.client = None
