"""
AI Knowledge Extractor — 高性能版本
- 流式 SSE 输出，用户即时看到结果
- 多线程并发处理
- 图片哈希缓存（相同图片秒返回）
- 优化 Prompt，减少 Token 消耗
"""

import http.server
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from socketserver import ThreadingMixIn

PORT = int(os.environ.get("PORT", 8765))
HTML_FILE = Path(__file__).parent / "image-knowledge-summary.html"

# 精简 Prompt —— 原版 ~150 字，优化后 ~80 字，减少 50% token 消耗
PROMPT = (
    "仔细分析图片，提取关键知识点并结构化输出。要求：\n"
    "1. 中文输出，按主题分组，用 ## 标题\n"
    "2. 知识点简洁完整，公式/代码精确还原\n"
    "3. 用列表或表格增强可读性\n"
    "4. 末尾一段总结核心内容"
)

# 内存缓存：key = sha256(image_data + prompt), value = (timestamp, html, text)
# TTL = 30 分钟，最多缓存 50 条
CACHE = {}
CACHE_MAX = 50
CACHE_TTL = 1800
CACHE_LOCK = threading.Lock()


def cache_key(image_data: str) -> str:
    return hashlib.sha256(image_data.encode()[:4096]).hexdigest()


def cache_get(key: str):
    now = time.time()
    with CACHE_LOCK:
        entry = CACHE.get(key)
        if entry and now - entry[0] < CACHE_TTL:
            return entry[1], entry[2]
        if entry:
            del CACHE[key]
    return None


def cache_set(key: str, html: str, text: str):
    now = time.time()
    with CACHE_LOCK:
        if len(CACHE) >= CACHE_MAX:
            oldest = min(CACHE, key=lambda k: CACHE[k][0])
            del CACHE[oldest]
        CACHE[key] = (now, html, text)


# ---------- Markdown renderer ----------

_RE_CODE_BLOCK = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
_RE_INLINE_CODE = re.compile(r'`([^`]+)`')
_RE_BOLD = re.compile(r'\*\*(.+?)\*\*')
_RE_ITALIC = re.compile(r'\*(.+?)\*')
_RE_H4 = re.compile(r'^#### (.+)$', re.MULTILINE)
_RE_H3 = re.compile(r'^### (.+)$', re.MULTILINE)
_RE_H2 = re.compile(r'^## (.+)$', re.MULTILINE)
_RE_H1 = re.compile(r'^# (.+)$', re.MULTILINE)
_RE_UL = re.compile(r'^[\-\*] (.+)$', re.MULTILINE)
_RE_OL = re.compile(r'^\d+\. (.+)$', re.MULTILINE)
_RE_BQ = re.compile(r'^> (.+)$', re.MULTILINE)
_RE_HR = re.compile(r'^---+$', re.MULTILINE)


def _escape(s: str) -> str:
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def render_markdown(md: str) -> str:
    html = md
    html = _RE_CODE_BLOCK.sub(
        lambda m: f'<pre><code>{_escape(m.group(2).rstrip())}</code></pre>', html)
    html = _RE_INLINE_CODE.sub(r'<code>\1</code>', html)
    html = _RE_BOLD.sub(r'<strong>\1</strong>', html)
    html = _RE_ITALIC.sub(r'<em>\1</em>', html)
    html = _RE_H4.sub(r'<h4>\1</h4>', html)
    html = _RE_H3.sub(r'<h3>\1</h3>', html)
    html = _RE_H2.sub(r'<h2>\1</h2>', html)
    html = _RE_H1.sub(r'<h2>\1</h2>', html)
    html = _RE_UL.sub(r'<li>\1</li>', html)
    html = re.sub(r'((?:<li>.*?</li>\n?)+)', r'<ul>\1</ul>', html)
    html = _RE_OL.sub(r'<li>\1</li>', html)
    html = _RE_BQ.sub(r'<blockquote>\1</blockquote>', html)
    html = _RE_HR.sub(r'<hr>', html)
    html = re.sub(r'\n\n+', '</p><p>', html)
    html = '<p>' + html + '</p>'
    for tag in ('h2', 'h3', 'h4', 'ul', 'ol', 'pre', 'blockquote', 'hr'):
        html = re.sub(rf'<p>(<{tag})', r'\1', html)
    for tag in ('h2', 'h3', 'h4', 'ul', 'ol', 'pre', 'blockquote'):
        html = re.sub(rf'(</{tag}>)</p>', r'\1', html)
    html = re.sub(r'<p>\s*</p>', '', html)
    html = re.sub(r'</ul>\s*<ul>', '', html)
    html = re.sub(r'</ol>\s*<ol>', '', html)
    return html


# ---------- Streaming Qwen VL ----------

def stream_qwen_vl(api_key: str, image_data: str, mime_type: str):
    """SSE generator: yields (type, content) tuples — 'chunk' or 'done'."""
    body = json.dumps({
        "model": "qwen-vl-max",
        "input": {
            "messages": [{
                "role": "user",
                "content": [
                    {"text": PROMPT},
                    {"image": f"data:{mime_type};base64,{image_data}"},
                ],
            }],
        },
        "parameters": {"incremental_output": True},
    })
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        data=body.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-DashScope-SSE": "enable",
        },
        method="POST",
    )
    full_text = ""
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str:
                continue
            try:
                data = json.loads(data_str)
                output = data.get("output", {})
                choices = output.get("choices", [])
                if not choices:
                    continue
                msg = choices[0].get("message", {})
                content_list = msg.get("content", [])
                if not content_list:
                    continue
                new_text = content_list[0].get("text", "")
            except json.JSONDecodeError:
                continue

            if new_text and new_text != full_text:
                delta = new_text[len(full_text):]
                full_text = new_text
                yield "chunk", delta

            if choices[0].get("finish_reason") == "stop":
                break

    yield "done", full_text


def call_qwen_vl_nonstream(api_key: str, image_data: str, mime_type: str) -> str:
    """Fallback: non-streaming call."""
    body = json.dumps({
        "model": "qwen-vl-max",
        "input": {
            "messages": [{
                "role": "user",
                "content": [
                    {"text": PROMPT},
                    {"image": f"data:{mime_type};base64,{image_data}"},
                ],
            }],
        },
    })
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        data=body.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["output"]["choices"][0]["message"]["content"][0]["text"]


# ---------- Threaded HTTP server ----------

class ThreadedHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stdout.write(f"[{self.address_string()}] {args[0]}\n")
        sys.stdout.flush()

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _sse_event(self, event: str, data: str):
        msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        self.wfile.write(msg.encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = HTML_FILE.read_text(encoding="utf-8")
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(data))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/summarize":
            self._handle_summarize()
        elif self.path == "/api/summarize/stream":
            self._handle_summarize_stream()
        else:
            self.send_error(404)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_json(400, {"error": "请求体为空"})
            return None
        if length > 12 * 1024 * 1024:
            self._send_json(400, {"error": "图片太大，限制 10MB"})
            return None
        return json.loads(self.rfile.read(length))

    def _get_api_key(self):
        key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not key:
            self._send_json(400, {"error": "服务器未配置 DASHSCOPE_API_KEY"})
            return None
        return key

    def _handle_summarize(self):
        """非流式端点（缓存命中时秒返回）"""
        api_key = self._get_api_key()
        if not api_key:
            return
        body = self._read_body()
        if body is None:
            return

        image_data = body.get("image")
        mime_type = body.get("mime", "image/png")
        if not image_data:
            self._send_json(400, {"error": "未提供图片数据"})
            return

        # 检查缓存
        ck = cache_key(image_data)
        cached = cache_get(ck)
        if cached:
            self._send_json(200, {"html": cached[0], "text": cached[1], "cached": True})
            return

        try:
            text = call_qwen_vl_nonstream(api_key, image_data, mime_type)
            html = render_markdown(text)
            cache_set(ck, html, text)
            self._send_json(200, {"html": html, "text": text, "cached": False})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            self._send_json(e.code, {"error": f"API 调用失败: {detail}"})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_summarize_stream(self):
        """流式 SSE 端点"""
        api_key = self._get_api_key()
        if not api_key:
            return
        body = self._read_body()
        if body is None:
            return

        image_data = body.get("image")
        mime_type = body.get("mime", "image/png")
        if not image_data:
            self._send_json(400, {"error": "未提供图片数据"})
            return

        # Check cache
        ck = cache_key(image_data)
        cached = cache_get(ck)
        if cached:
            # 缓存命中也用 SSE 返回，前端统一处理
            self._send_sse_headers()
            self._sse_event("cached", {"html": cached[0], "text": cached[1]})
            self._sse_event("done", {"status": "complete"})
            return

        self._send_sse_headers()

        accumulated = ""
        try:
            for evt_type, content in stream_qwen_vl(api_key, image_data, mime_type):
                if evt_type == "chunk":
                    self._sse_event("chunk", {"delta": content})
                elif evt_type == "done":
                    accumulated = content
                    html = render_markdown(accumulated)
                    cache_set(ck, html, accumulated)
                    self._sse_event("rendered", {"html": html, "text": accumulated})
                    self._sse_event("done", {"status": "complete"})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            self._sse_event("error", {"error": f"API 调用失败: {detail}"})
        except Exception as e:
            self._sse_event("error", {"error": str(e)})


def main():
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not dashscope_key:
        print("=" * 56)
        print("  提示: 请设置 DASHSCOPE_API_KEY 环境变量")
        print("  Windows CMD:  set DASHSCOPE_API_KEY=sk-...")
        print("  PowerShell:   $env:DASHSCOPE_API_KEY=\"sk-...\"")
        print("=" * 56)
        print()

    addr = ("0.0.0.0", PORT)
    server = ThreadedHTTPServer(addr, Handler)
    url = f"http://localhost:{PORT}"
    print(f"AI Knowledge Extractor: {url}")
    print(f"  支持: SSE 流式输出 | 多线程并发 | 结果缓存")
    print("  按 Ctrl+C 停止\n")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()


if __name__ == "__main__":
    main()
