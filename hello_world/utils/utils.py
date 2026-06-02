from anthropic.types import Usage


def ok(content) -> dict:
    """Success result for a tool."""
    return {"content": content, "is_error": False}


def err(message: str) -> dict:
    """Error result for a tool.

    `message` is shown to the LLM — say what went wrong AND what to do next.
    """
    return {"content": message, "is_error": True}


def usage_add(usage_1: Usage, usage_2: Usage) -> Usage:
    cache_creation_5m = (0 if usage_1.cache_creation is None else usage_1.cache_creation.ephemeral_5m_input_tokens) + (
        0 if usage_2.cache_creation is None else usage_2.cache_creation.ephemeral_5m_input_tokens
    )

    cache_creation_input_tokens = (
        0 if usage_1.cache_creation_input_tokens is None else usage_1.cache_creation_input_tokens
    ) + (0 if usage_2.cache_creation_input_tokens is None else usage_2.cache_creation_input_tokens)

    cache_read_input_tokens = (0 if usage_1.cache_read_input_tokens is None else usage_1.cache_read_input_tokens) + (
        0 if usage_2.cache_read_input_tokens is None else usage_2.cache_read_input_tokens
    )

    return Usage(
        cache_creation={"ephemeral_5m_input_tokens": cache_creation_5m, "ephemeral_1h_input_tokens": 0},
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        input_tokens=usage_1.input_tokens + usage_2.input_tokens,
        output_tokens=usage_1.output_tokens + usage_2.output_tokens,
        output_tokens_details=None,
        server_tool_use=None,
        service_tier=None,
        inference_geo=None,
    )


def print_usage(usage: Usage) -> None:
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
    if usage.cache_creation_input_tokens is not None:
        print(f"Cache creation input tokens: {usage.cache_creation_input_tokens}")
    if usage.cache_read_input_tokens is not None:
        print(f"Cache read input tokens: {usage.cache_read_input_tokens}")
    if usage.cache_creation is not None:
        print(f"Cache creation ephemeral 5m input tokens: {usage.cache_creation.ephemeral_5m_input_tokens}")
        print(f"Cache creation ephemeral 1h input tokens: {usage.cache_creation.ephemeral_1h_input_tokens}")
