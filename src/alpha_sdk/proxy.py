"""Async proxy server for request interception.

Runs an aiohttp server in the same event loop as the SDK client:
1. Receives requests from Claude Agent SDK (via Claude Code subprocess)
2. Checks for canary in the last user message content block
3. If canary present: weave (inject Alpha's soul), strip canary
4. If canary absent: pass through unchanged (internal SDK call)
5. Forwards to Anthropic
6. Streams responses back

Because it's async in the same event loop, all spans share trace context.
One turn → one span → everything nested inside.
"""

import asyncio
import json
import socket
from contextlib import nullcontext
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import AlphaClient

import httpx
import logfire
from aiohttp import web

from .canary import is_canary_block
from .attributes import request_attributes

ANTHROPIC_API_URL = "https://api.anthropic.com"

# Headers to forward from incoming request (SDK → Anthropic)
# These handle authentication - DO NOT add our own API keys!
FORWARD_HEADERS = [
    "authorization",      # OAuth Bearer token (Claude Max)
    "x-api-key",          # API key auth (fallback)
    "anthropic-version",
    "anthropic-beta",
    "content-type",
]

# Headers to skip when forwarding response (hop-by-hop)
SKIP_RESPONSE_HEADERS = {
    "content-encoding",
    "transfer-encoding",
    "connection",
    "keep-alive",
}


def _find_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class AlphaProxy:
    """Async proxy server for intercepting SDK requests.

    Runs in the same event loop as the caller, enabling shared trace context.
    Has direct access to AlphaClient state (system prompt, etc.) since they
    share the same process and event loop.

    Usage:
        proxy = AlphaProxy(alpha_client=client)
        await proxy.start()

        os.environ["ANTHROPIC_BASE_URL"] = proxy.base_url

        # Before each turn, set the trace context so proxy spans nest properly
        proxy.set_trace_context(logfire.get_context())

        # ... use SDK ...

        await proxy.stop()
    """

    def __init__(self, alpha_client: "AlphaClient"):  # noqa: F821 - forward reference
        """Initialize the proxy.

        Args:
            alpha_client: The AlphaClient instance (for accessing system prompt, etc.)
        """
        self.alpha_client = alpha_client

        self._port: int | None = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._trace_context: dict | None = None

    async def start(self) -> int:
        """Start the proxy server.

        Returns:
            The port number the server is listening on.
        """
        self._port = _find_free_port()

        # Create long-lived httpx client for forwarding requests
        self._http_client = httpx.AsyncClient(timeout=300.0)

        # Build aiohttp app
        self._app = web.Application()
        self._app.router.add_route("*", "/{path:.*}", self._handle_request)

        # Start server
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", self._port)
        await self._site.start()

        logfire.debug(f"Alpha proxy started on port {self._port}")
        return self._port

    async def stop(self) -> None:
        """Stop the proxy server."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        if self._runner:
            await self._runner.cleanup()
            self._runner = None

        self._site = None
        self._app = None

        logfire.debug("Alpha proxy stopped")

    @property
    def base_url(self) -> str:
        """Get the base URL for this proxy."""
        if self._port is None:
            raise RuntimeError("Proxy not started")
        return f"http://127.0.0.1:{self._port}"

    @property
    def port(self) -> int | None:
        """Get the port number."""
        return self._port

    def set_trace_context(self, ctx: dict) -> None:
        """Set the trace context for request handlers.

        Call this before each turn so proxy spans nest under the turn span.

        Args:
            ctx: Trace context from logfire.get_context()
        """
        self._trace_context = ctx

    def _check_and_strip_canary(self, body: dict) -> bool:
        """Check if the request has our canary, and strip it if so.

        The canary is the last content block in the last user message.
        If found, we remove it (it's just routing metadata) and return True.

        Args:
            body: The request body (modified in place if canary found)

        Returns:
            True if canary was found and stripped, False otherwise
        """
        messages = body.get("messages", [])
        if not messages:
            return False

        # Find the last user message
        last_user_msg = None
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                last_user_msg = msg
                last_user_idx = i

        if last_user_msg is None:
            return False

        # Check if content is a list (could be string for simple messages)
        content = last_user_msg.get("content")
        if not isinstance(content, list) or len(content) == 0:
            return False

        # Check the last content block
        last_block = content[-1]
        if is_canary_block(last_block):
            # Strip the canary block - it's just routing metadata
            content.pop()
            return True

        return False

    async def _handle_request(self, request: web.Request) -> web.StreamResponse:
        """Handle incoming requests from the SDK."""
        path = "/" + request.match_info.get("path", "")

        # Health check
        if request.method == "GET" and path == "/health":
            return web.Response(text="ok")

        # Only handle POST to messages endpoints
        if request.method != "POST":
            return web.Response(status=404, text="Not found")

        # Attach trace context so spans nest under the current turn
        context_manager = (
            logfire.attach_context(self._trace_context)
            if self._trace_context
            else nullcontext()
        )

        with context_manager:
            with logfire.span(
                "proxy.forward",
                path=path,
                method=request.method,
            ) as span:
                try:
                    return await self._forward_request(request, path, span)
                except Exception as e:
                    logfire.error(f"Proxy error: {e}")
                    span.set_attribute("error", str(e))
                    return web.Response(status=500, text=str(e))

    async def _forward_request(
        self,
        request: web.Request,
        path: str,
        span: logfire.LogfireSpan,
    ) -> web.StreamResponse:
        """Transform and forward a request to Anthropic."""
        # Read request body
        body_bytes = await request.read()

        try:
            body = await request.json()
        except Exception:
            # Not JSON - forward as-is (shouldn't happen for messages API)
            body = None

        # Check for canary in the last user message
        should_weave = False
        if body is not None:
            should_weave = self._check_and_strip_canary(body)
            span.set_attribute("has_canary", should_weave)

        # Inject system prompt if canary was present
        if should_weave:
            system_prompt = self.alpha_client._system_prompt
            if system_prompt:
                # Preserve SDK's billing header if present
                existing_system = body.get("system")
                if isinstance(existing_system, list) and len(existing_system) >= 1:
                    # SDK sends: [0]=billing header, keep it
                    billing_header = existing_system[0]
                    body["system"] = [billing_header] + system_prompt
                    span.set_attribute("merge_mode", "keep_billing_header")
                else:
                    body["system"] = system_prompt
                    span.set_attribute("merge_mode", "replace")

                span.set_attribute("woven", True)
                span.set_attribute("system_blocks", len(system_prompt))
                logfire.debug(f"Injected system prompt ({len(system_prompt)} blocks)")
            else:
                logfire.warn("Canary present but no system prompt on client")
                span.set_attribute("woven", False)
        elif body is not None:
            span.set_attribute("woven", False)
            logfire.debug("Pass-through (no canary)")

        # Add gen_ai.* attributes for Logfire Model Run panel
        # Note: response attributes (tokens, finish_reason) would require parsing
        # the SSE stream, which we don't do—we just pipe chunks through.
        # The SDK gets usage info from ResultMessage anyway.
        if body is not None:
            # Extract session_id from the messages if available (canary had it)
            session_id = None  # TODO: extract from envelope if needed
            for attr_name, attr_value in request_attributes(body, session_id).items():
                span.set_attribute(attr_name, attr_value)

        # Build headers - forward auth headers from SDK
        headers = {}
        for header_name in FORWARD_HEADERS:
            value = request.headers.get(header_name)
            if value:
                headers[header_name] = value

        # Ensure content-type
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        # Forward to Anthropic
        url = f"{ANTHROPIC_API_URL}{path}"

        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")

        # Use streaming to forward response in real-time
        async with self._http_client.stream(
            "POST",
            url,
            json=body if body is not None else None,
            content=body_bytes if body is None else None,
            headers=headers,
        ) as response:
            span.set_attribute("status_code", response.status_code)

            # Create streaming response
            resp = web.StreamResponse(status=response.status_code)

            # Forward response headers
            for key, value in response.headers.items():
                if key.lower() not in SKIP_RESPONSE_HEADERS:
                    resp.headers[key] = value

            await resp.prepare(request)

            # Stream response body, capturing SSE events for response attributes
            output_tokens = 0
            input_tokens = 0
            stop_reason = None
            response_model = None

            # Track output content as it streams
            output_text_parts: list[str] = []
            tool_calls: list[dict] = []
            current_tool_index: int = -1

            async for chunk in response.aiter_bytes():
                await resp.write(chunk)

                # Parse SSE events for usage stats and output content
                # Format: "event: message_delta\ndata: {...}\n\n"
                try:
                    chunk_str = chunk.decode("utf-8", errors="ignore")
                    for line in chunk_str.split("\n"):
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            event_type = data.get("type")

                            if event_type == "message_start":
                                # Initial message has input tokens and model
                                msg = data.get("message", {})
                                usage = msg.get("usage", {})
                                input_tokens = usage.get("input_tokens", 0)
                                response_model = msg.get("model")

                            elif event_type == "content_block_start":
                                # New content block starting
                                block = data.get("content_block", {})
                                if block.get("type") == "tool_use":
                                    tool_calls.append({
                                        "id": block.get("id", ""),
                                        "name": block.get("name", ""),
                                        "input_json": "",
                                    })
                                    current_tool_index = len(tool_calls) - 1

                            elif event_type == "content_block_delta":
                                # Content streaming in
                                delta = data.get("delta", {})
                                delta_type = delta.get("type")

                                if delta_type == "text_delta":
                                    output_text_parts.append(delta.get("text", ""))

                                elif delta_type == "input_json_delta":
                                    # Tool call input streaming (JSON fragments)
                                    if current_tool_index >= 0:
                                        tool_calls[current_tool_index]["input_json"] += delta.get("partial_json", "")

                            elif event_type == "message_delta":
                                # Final delta has output tokens and stop reason
                                delta = data.get("delta", {})
                                usage = data.get("usage", {})
                                if usage.get("output_tokens"):
                                    output_tokens = usage["output_tokens"]
                                if delta.get("stop_reason"):
                                    stop_reason = delta["stop_reason"]
                except Exception:
                    pass  # Don't fail on parse errors, just miss the attributes

            # Build gen_ai.output.messages
            output_parts: list[dict] = []
            if output_text_parts:
                output_parts.append({
                    "type": "text",
                    "content": "".join(output_text_parts),
                })
            for tool in tool_calls:
                try:
                    tool_input = json.loads(tool["input_json"]) if tool["input_json"] else {}
                except json.JSONDecodeError:
                    tool_input = {}
                output_parts.append({
                    "type": "tool_call",
                    "id": tool["id"],
                    "name": tool["name"],
                    "arguments": tool_input,
                })

            output_message = {"role": "assistant", "parts": output_parts}
            if stop_reason:
                output_message["finish_reason"] = stop_reason

            # Set response attributes
            if input_tokens:
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            if output_tokens:
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            if stop_reason:
                span.set_attribute("gen_ai.response.finish_reasons", [stop_reason])
            if response_model:
                span.set_attribute("gen_ai.response.model", response_model)
            if output_parts:
                span.set_attribute("gen_ai.output.messages", json.dumps([output_message]))

            # Determine output type
            has_tool = any(p.get("type") == "tool_call" for p in output_parts)
            has_text = any(p.get("type") == "text" for p in output_parts)
            if has_tool:
                span.set_attribute("gen_ai.output.type", "json")
            elif has_text:
                span.set_attribute("gen_ai.output.type", "text")

            await resp.write_eof()
            return resp
