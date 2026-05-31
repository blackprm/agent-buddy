"""Web tools migrated from Claude Code: WebFetch and WebSearch.

The implementation intentionally keeps the same user-facing tool names and
schemas as Claude Code while using only the Python standard library by default.
"""
from __future__ import annotations

import asyncio
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable

from agent_core.tools.base import ToolContext, ToolResult


MAX_HTTP_CONTENT_LENGTH = 10 * 1024 * 1024
MAX_MARKDOWN_LENGTH = 100_000
FETCH_TIMEOUT_SECONDS = 60
MAX_REDIRECTS = 10
USER_AGENT = "my-agent-core WebFetch/1.0 (+https://github.com/anthropics/claude-code-compatible)"

FetchCallable = Callable[[str], Awaitable[dict[str, Any]]]
SearchCallable = Callable[[str, list[str] | None, list[str] | None], Awaitable[list[dict[str, str]]]]


@dataclass(slots=True)
class _FetchedPage:
    url: str
    code: int
    code_text: str
    content_type: str
    body: bytes


@dataclass(slots=True)
class _RedirectInfo:
    original_url: str
    redirect_url: str
    status_code: int


class WebFetchTool:
    name = "WebFetch"
    description = """Fetches content from a specified URL and returns text content for analysis.

IMPORTANT: WebFetch WILL FAIL for authenticated or private URLs. Before using this tool,
check if the URL points to an authenticated service (e.g. Google Docs, Confluence, Jira,
GitHub private repos). If so, prefer a specialized authenticated integration/MCP tool.

Input uses Claude Code-compatible fields:
- url: fully-qualified http(s) URL to fetch
- prompt: what information to extract/analyze from the fetched content

HTTP URLs are upgraded to HTTPS for public hosts where practical. Cross-host redirects are
not followed automatically; the tool returns the redirect URL and asks the model to call
WebFetch again for the new host."""
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch content from"},
            "prompt": {"type": "string", "description": "The prompt to run on the fetched content"},
        },
        "required": ["url", "prompt"],
    }
    is_concurrency_safe = True
    should_defer = False

    def __init__(self, *, fetcher: FetchCallable | None = None) -> None:
        self._fetcher = fetcher

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        url = str(tool_input.get("url") or tool_input.get("uri") or "").strip()
        prompt = str(tool_input.get("prompt") or "").strip()
        if not url:
            return ToolResult(content="WebFetch failed: missing required field 'url'", is_error=True)
        if not prompt:
            return ToolResult(content="WebFetch failed: missing required field 'prompt'", is_error=True)
        if not _validate_url(url):
            return ToolResult(content=f"WebFetch failed: invalid or unsupported URL: {url}", is_error=True)

        start = time.perf_counter()
        try:
            if self._fetcher:
                raw = await self._fetcher(url)
                page_or_redirect = _coerce_fetcher_result(url, raw)
            else:
                page_or_redirect = await asyncio.to_thread(_fetch_with_permitted_redirects, url)
        except Exception as exc:
            return ToolResult(content=f"WebFetch failed for {url}: {exc}", is_error=True, metadata={"url": url})

        duration_ms = int((time.perf_counter() - start) * 1000)
        if isinstance(page_or_redirect, _RedirectInfo):
            status_text = _status_text(page_or_redirect.status_code)
            message = (
                "REDIRECT DETECTED: The URL redirects to a different host.\n\n"
                f"Original URL: {page_or_redirect.original_url}\n"
                f"Redirect URL: {page_or_redirect.redirect_url}\n"
                f"Status: {page_or_redirect.status_code} {status_text}\n\n"
                "To complete the request, call WebFetch again with the redirected URL."
            )
            return ToolResult(
                content=message,
                metadata={
                    "url": url,
                    "redirectUrl": page_or_redirect.redirect_url,
                    "code": page_or_redirect.status_code,
                    "codeText": status_text,
                    "durationMs": duration_ms,
                    "bytes": len(message.encode("utf-8")),
                },
            )

        text = _page_to_text(page_or_redirect)
        truncated = _truncate(text, MAX_MARKDOWN_LENGTH)
        result = (
            f"Fetched: {page_or_redirect.url}\n"
            f"Status: {page_or_redirect.code} {page_or_redirect.code_text}\n"
            f"Content-Type: {page_or_redirect.content_type or 'unknown'}\n"
            f"Bytes: {len(page_or_redirect.body)}\n\n"
            f"Prompt: {prompt}\n\n"
            "Content:\n"
            f"{truncated}"
        )
        return ToolResult(
            content=result,
            metadata={
                "url": page_or_redirect.url,
                "requestedUrl": url,
                "code": page_or_redirect.code,
                "codeText": page_or_redirect.code_text,
                "contentType": page_or_redirect.content_type,
                "bytes": len(page_or_redirect.body),
                "durationMs": duration_ms,
                "truncated": len(text) > MAX_MARKDOWN_LENGTH,
            },
        )


class WebSearchTool:
    name = "WebSearch"
    description = """Search the web for current information and return citation-ready results.

Use this when the user asks for recent information, documentation, or facts that may have
changed. Supports optional domain filters:
- allowed_domains: only include results from these domains
- blocked_domains: exclude results from these domains

After using WebSearch, cite relevant result URLs in the final answer."""
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2, "description": "The search query to use"},
            "allowed_domains": {"type": "array", "items": {"type": "string"}, "description": "Only include search results from these domains"},
            "blocked_domains": {"type": "array", "items": {"type": "string"}, "description": "Never include search results from these domains"},
        },
        "required": ["query"],
    }
    is_concurrency_safe = True
    should_defer = False

    def __init__(self, *, searcher: SearchCallable | None = None) -> None:
        self._searcher = searcher

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        query = str(tool_input.get("query") or "").strip()
        if len(query) < 2:
            return ToolResult(content="WebSearch failed: query must contain at least 2 characters", is_error=True)
        allowed = _string_list(tool_input.get("allowed_domains"))
        blocked = _string_list(tool_input.get("blocked_domains"))
        if allowed and blocked:
            overlap = set(_canonical_domain(d) for d in allowed) & set(_canonical_domain(d) for d in blocked)
            if overlap:
                return ToolResult(content=f"WebSearch failed: domains cannot be both allowed and blocked: {', '.join(sorted(overlap))}", is_error=True)

        start = time.perf_counter()
        try:
            if self._searcher:
                results = await self._searcher(query, allowed, blocked)
            else:
                results = await asyncio.to_thread(_default_web_search, query, allowed, blocked)
        except Exception as exc:
            return ToolResult(content=f"WebSearch failed for query {query!r}: {exc}", is_error=True, metadata={"query": query})

        filtered = _filter_search_results(results, allowed, blocked)
        duration_seconds = round(time.perf_counter() - start, 3)
        if not filtered:
            return ToolResult(
                content=f"No web search results found for: {query}",
                metadata={"query": query, "results": [], "durationSeconds": duration_seconds},
            )

        lines = [f"Search results for: {query}", ""]
        for i, result in enumerate(filtered, start=1):
            title = result.get("title") or result.get("url") or "Untitled"
            url = result.get("url") or ""
            snippet = result.get("snippet") or ""
            lines.append(f"{i}. {title}")
            lines.append(f"   URL: {url}")
            if snippet:
                lines.append(f"   Snippet: {snippet}")
        lines.extend(["", "When answering, cite the relevant URLs from these results."])
        return ToolResult(
            content="\n".join(lines),
            metadata={"query": query, "results": filtered, "durationSeconds": duration_seconds},
        )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        if tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        joined = html.unescape("".join(self.parts))
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in joined.splitlines()]
        return "\n".join(line for line in lines if line)


def _validate_url(url: str) -> bool:
    if len(url) > 2000:
        return False
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.hostname or parsed.username or parsed.password:
        return False
    return True


def _fetch_with_permitted_redirects(url: str, depth: int = 0) -> _FetchedPage | _RedirectInfo:
    if depth > MAX_REDIRECTS:
        raise RuntimeError(f"too many redirects (>{MAX_REDIRECTS})")
    request_url = _quote_url_for_request(_upgrade_http_url(url))
    req = urllib.request.Request(request_url, headers={"Accept": "text/markdown, text/html, text/plain, */*", "User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(_NoRedirectHandler)
    try:
        with opener.open(req, timeout=FETCH_TIMEOUT_SECONDS) as response:
            body = response.read(MAX_HTTP_CONTENT_LENGTH + 1)
            if len(body) > MAX_HTTP_CONTENT_LENGTH:
                raise RuntimeError("response exceeded 10MB limit")
            return _FetchedPage(
                url=response.geturl(),
                code=int(response.status),
                code_text=str(response.reason or "OK"),
                content_type=response.headers.get("content-type", ""),
                body=body,
            )
    except urllib.error.HTTPError as exc:
        if exc.code in {301, 302, 307, 308}:
            location = exc.headers.get("Location")
            if not location:
                raise RuntimeError("redirect missing Location header")
            redirect_url = urllib.parse.urljoin(request_url, location)
            if _is_permitted_redirect(request_url, redirect_url):
                return _fetch_with_permitted_redirects(redirect_url, depth + 1)
            return _RedirectInfo(original_url=request_url, redirect_url=redirect_url, status_code=exc.code)
        raise RuntimeError(f"HTTP {exc.code}: {exc.reason}") from exc


def _upgrade_http_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme == "http" and host not in {"localhost", "127.0.0.1", "::1"}:
        parsed = parsed._replace(scheme="https")
        return urllib.parse.urlunparse(parsed)
    return url


def _quote_url_for_request(url: str) -> str:
    """Return an ASCII-only URL suitable for urllib Request.

    urllib's HTTP layer encodes the request target as ASCII. Browser-style URLs
    often contain raw unicode query/path characters (for example Baidu search
    URLs with Chinese keywords), so quote each URL component before opening it.
    """
    parsed = urllib.parse.urlsplit(url)
    hostname = parsed.hostname.encode("idna").decode("ascii") if parsed.hostname else ""
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    path = urllib.parse.quote(parsed.path or "/", safe="/%:@!$&'()*+,;=")
    query = urllib.parse.quote(parsed.query, safe="=&?/:;+,%@!$'()*[]")
    fragment = urllib.parse.quote(parsed.fragment, safe="=&?/:;+,%@!$'()*[]")
    return urllib.parse.urlunsplit((parsed.scheme, netloc, path, query, fragment))


def _is_permitted_redirect(original_url: str, redirect_url: str) -> bool:
    original = urllib.parse.urlparse(original_url)
    redirect = urllib.parse.urlparse(redirect_url)
    if original.scheme != redirect.scheme or original.port != redirect.port:
        return False
    if redirect.username or redirect.password:
        return False
    strip = lambda value: (value or "").removeprefix("www.")
    return strip(original.hostname) == strip(redirect.hostname)


def _page_to_text(page: _FetchedPage) -> str:
    raw_text = page.body.decode("utf-8", errors="replace")
    content_type = page.content_type.lower()
    if "html" not in content_type:
        return raw_text.strip()
    parser = _TextExtractor()
    parser.feed(raw_text)
    return parser.text()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[Content truncated due to length...]"


def _coerce_fetcher_result(url: str, raw: dict[str, Any]) -> _FetchedPage | _RedirectInfo:
    if raw.get("type") == "redirect":
        return _RedirectInfo(
            original_url=str(raw.get("originalUrl") or url),
            redirect_url=str(raw.get("redirectUrl") or raw.get("url") or ""),
            status_code=int(raw.get("statusCode") or raw.get("code") or 302),
        )
    body = raw.get("body", raw.get("content", ""))
    if isinstance(body, str):
        body_bytes = body.encode("utf-8")
    else:
        body_bytes = bytes(body)
    return _FetchedPage(
        url=str(raw.get("url") or url),
        code=int(raw.get("code") or raw.get("status") or 200),
        code_text=str(raw.get("codeText") or raw.get("reason") or "OK"),
        content_type=str(raw.get("contentType") or raw.get("content_type") or "text/plain"),
        body=body_bytes,
    )


def _status_text(status_code: int) -> str:
    return {301: "Moved Permanently", 302: "Found", 307: "Temporary Redirect", 308: "Permanent Redirect"}.get(status_code, "Redirect")


def _string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _canonical_domain(domain: str) -> str:
    return domain.lower().removeprefix("www.").strip()


def _filter_search_results(results: list[dict[str, str]], allowed: list[str] | None, blocked: list[str] | None) -> list[dict[str, str]]:
    allowed_set = {_canonical_domain(d) for d in allowed or []}
    blocked_set = {_canonical_domain(d) for d in blocked or []}
    filtered: list[dict[str, str]] = []
    seen: set[str] = set()
    for result in results:
        url = result.get("url") or ""
        host = _canonical_domain(urllib.parse.urlparse(url).hostname or "")
        if not url or url in seen:
            continue
        if allowed_set and not any(host == d or host.endswith("." + d) for d in allowed_set):
            continue
        if blocked_set and any(host == d or host.endswith("." + d) for d in blocked_set):
            continue
        seen.add(url)
        filtered.append({"title": result.get("title") or url, "url": url, "snippet": result.get("snippet") or ""})
    return filtered[:10]


def _default_web_search(query: str, allowed: list[str] | None, blocked: list[str] | None) -> list[dict[str, str]]:
    provider = (os.getenv("AGENT_WEB_SEARCH_PROVIDER") or "").strip().lower()
    endpoint = os.getenv("AGENT_WEB_SEARCH_ENDPOINT")
    if provider == "searxng":
        return _search_searxng(endpoint or "http://127.0.0.1:18888", query, allowed, blocked)
    if endpoint:
        return _search_via_json_endpoint(endpoint, query, allowed, blocked)
    return _search_duckduckgo_html(query)


def _search_searxng(endpoint: str, query: str, allowed: list[str] | None, blocked: list[str] | None) -> list[dict[str, str]]:
    base = endpoint.rstrip("/")
    search_url = base if base.endswith("/search") else base + "/search"
    params: dict[str, str] = {"q": query, "format": "json"}
    if allowed:
        params["site"] = " OR ".join(f"site:{domain}" for domain in allowed)
        params["q"] = query + " " + params["site"]
    url = search_url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read(MAX_HTTP_CONTENT_LENGTH).decode("utf-8"))
    raw_results = payload.get("results", []) if isinstance(payload, dict) else []
    results: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url_value = str(item.get("url") or "").strip()
        if not url_value:
            continue
        results.append({
            "title": str(item.get("title") or url_value),
            "url": url_value,
            "snippet": str(item.get("content") or item.get("snippet") or ""),
            "engine": str(item.get("engine") or ""),
        })
    return results


def _search_via_json_endpoint(endpoint: str, query: str, allowed: list[str] | None, blocked: list[str] | None) -> list[dict[str, str]]:
    params = {"q": query}
    if allowed:
        params["allowed_domains"] = ",".join(allowed)
    if blocked:
        params["blocked_domains"] = ",".join(blocked)
    url = endpoint + ("&" if "?" in endpoint else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read(MAX_HTTP_CONTENT_LENGTH).decode("utf-8"))
    raw_results = payload.get("results", payload if isinstance(payload, list) else [])
    return [r for r in raw_results if isinstance(r, dict)]


def _search_duckduckgo_html(query: str) -> list[dict[str, str]]:
    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_SECONDS) as response:
        html_text = response.read(MAX_HTTP_CONTENT_LENGTH).decode("utf-8", errors="replace")
    return _parse_duckduckgo_results(html_text)


def _parse_duckduckgo_results(html_text: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    pattern = re.compile(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S)
    for match in pattern.finditer(html_text):
        href = html.unescape(match.group(1))
        title = re.sub(r"<[^>]+>", "", match.group(2))
        title = html.unescape(re.sub(r"\s+", " ", title)).strip()
        parsed = urllib.parse.urlparse(href)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            qs = urllib.parse.parse_qs(parsed.query)
            href = qs.get("uddg", [href])[0]
        if title and href.startswith(("http://", "https://")):
            results.append({"title": title, "url": href, "snippet": ""})
    return results
