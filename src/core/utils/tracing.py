try:
    from langfuse.llama_index import LlamaIndexCallbackHandler
    has_langfuse = True
except ModuleNotFoundError:
    import warnings
    warnings.warn("Langfuse is not install, will not trace!")
    has_langfuse = False
from llama_index.core.callbacks import CallbackManager
from llama_index.core import Settings

from .settings import Settings as AppSettings


def setup_tracing(settings: AppSettings):
    if has_langfuse:
        callback_handler = LlamaIndexCallbackHandler(
            public_key=settings.tracing.public_key,
            secret_key=settings.tracing.secret_key,
            host=settings.tracing.host,
            user_id=settings.tracing.user_id
        )
        Settings.callback_manager = CallbackManager([callback_handler])
