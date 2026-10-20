class LLMTimeoutError(Exception):
    pass


class LLMProviderError(Exception):
    """The LLM backend rejected the request."""


class EmbeddingRateLimitError(LLMProviderError):
    """An embedding request must be deferred until the provider window ends."""

    def __init__(self, retry_after: float) -> None:
        self.retry_after = float(retry_after)
        self.full_rebuild = False
        # Keep Exception.args constructor-compatible for Celery/pickle. A
        # formatted string here would be passed back into this __init__ during
        # worker deserialization and fail numeric formatting.
        super().__init__(self.retry_after)

    def __str__(self) -> str:
        return f"Embedding provider requested retry after {self.retry_after:g}s"


class EmbeddingCapacityDeferred(EmbeddingRateLimitError):
    """Local pacing deferred a request before contacting the provider."""
