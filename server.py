"""
AI Knowledge Extractor v3 — 产品级后端
- 流式 SSE + 结构化提取
- 历史记录持久化
- 多格式导出
- 智能缓存
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
import uuid
from pathlib import Path
from socketserver import ThreadingMixIn

PORT = int(os.environ.get("PORT", 8765))
HTML_FILE = Path(__file__).parent / "index.html"
DATA_FILE = Path(__file__).parent / ".extractions.json"

PROMPT = (
    "分析这张图片，用如下 JSON 格式返回（只返回 JSON，不要其他文字）：\n"
    '{\n'
    '  "title": "图片主题（简短）",\n'
    '  "entities": [{"name": "实体名", "type": "概念/人物/事件/技术/其他", "weight": 1-10}],\n'
    '  "relations": [{"from": "实体A", "to": "实体B", "label": "关系描述"}],\n'
    '  "sections": [{"heading": "标题", "content": "要点（markdown）"}],\n'
    '  "summary": "一句话总结"\n'
    '}\n'
    "要求：中文输出，entities 至少 3 个，relations 至少 2 条，sections 至少 2 个。"
)

# ---------- Cache ----------
CACHE = {}
CACHE_LOCK = threading.Lock()
CACHE_MAX = 50
CACHE_TTL = 1800

# ---------- History ----------
HISTORY_LOCK = threading.Lock()
MAX_HISTORY = 20


def load_history():
    try:
        if DATA_FILE.exists():
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def save_history(entries):
    try:
        DATA_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def add_history(entry):
    with HISTORY_LOCK:
        entries = load_history()
        entries.insert(0, entry)
        if len(entries) > MAX_HISTORY:
            entries = entries[:MAX_HISTORY]
        save_history(entries)


def get_history(limit=10):
    with HISTORY_LOCK:
        entries = load_history()
    return entries[:limit]


# ---------- Cache helpers ----------
def cache_key(img: str) -> str:
    return hashlib.sha256(img.encode()[:4096]).hexdigest()


def cache_get(key: str):
    now = time.time()
    with CACHE_LOCK:
        e = CACHE.get(key)
        if e and now - e[0] < CACHE_TTL:
            return e[1]
        if e:
            del CACHE[key]
    return None


def cache_set(key: str, data: dict):
    now = time.time()
    with CACHE_LOCK:
        if len(CACHE) >= CACHE_MAX:
            oldest = min(CACHE, key=lambda k: CACHE[k][0])
            del CACHE[oldest]
        CACHE[key] = (now, data)


# ---------- Markdown ----------
_RE_CODE = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
_RE_IC = re.compile(r'`([^`]+)`')
_RE_B = re.compile(r'\*\*(.+?)\*\*')
_RE_I = re.compile(r'\*(.+?)\*')
_RE_H = {n: re.compile(rf'^{"#"*n} (.+)$', re.MULTILINE) for n in (1,2,3,4)}
_RE_UL = re.compile(r'^[\-\*] (.+)$', re.MULTILINE)
_RE_BQ = re.compile(r'^> (.+)$', re.MULTILINE)


def _esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def render_markdown(md: str) -> str:
    html = md
    html = _RE_CODE.sub(lambda m: f'<pre><code>{_esc(m.group(2).rstrip())}</code></pre>', html)
    html = _RE_IC.sub(r'<code>\1</code>', html)
    html = _RE_B.sub(r'<strong>\1</strong>', html)
    html = _RE_I.sub(r'<em>\1</em>', html)
    for n in (4,3,2,1):
        html = _RE_H[n].sub(rf'<h{n}>\1</h{n}>', html)
    html = _RE_UL.sub(r'<li>\1</li>', html)
    html = re.sub(r'((?:<li>.*?</li>\n?)+)', r'<ul>\1</ul>', html)
    html = _RE_BQ.sub(r'<blockquote>\1</blockquote>', html)
    html = re.sub(r'\n\n+', '</p><p>', html)
    html = '<p>' + html + '</p>'
    for t in ('h2','h3','h4','ul','ol','pre','blockquote'):
        html = re.sub(rf'<p>(<{t})', r'\1', html)
        html = re.sub(rf'(</{t}>)</p>', r'\1', html)
    html = re.sub(r'<p>\s*</p>', '', html)
    html = re.sub(r'</ul>\s*<ul>', '', html)
    return html


# ---------- LLM Call ----------
def stream_qwen_vl(api_key: str, image_b64: str, mime: str):
    """SSE generator yielding (type, data) tuples."""
    body = json.dumps({
        "model": "qwen-vl-max",
        "input": {
            "messages": [{
                "role": "user",
                "content": [
                    {"text": PROMPT},
                    {"image": f"data:{mime};base64,{image_b64}"},
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
    full = ""
    seen = set()
    phase = 0

    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            ds = line[5:].strip()
            if not ds:
                continue
            try:
                d = json.loads(ds)
                choices = d.get("output", {}).get("choices", [])
                if not choices:
                    continue
                msg = choices[0].get("message", {})
                cl = msg.get("content", [])
                if not cl:
                    continue
                txt = cl[0].get("text", "")
            except json.JSONDecodeError:
                continue

            if txt and txt != full:
                delta = txt[len(full):]
                full = txt
                yield "chunk", {"delta": delta}

                # Phase detection: as more data arrives, signal progress
                new_phase = phase
                if "entities" in full and "relations" not in full:
                    new_phase = 1
                elif "relations" in full and "sections" not in full:
                    new_phase = 2
                elif "sections" in full and "summary" not in full:
                    new_phase = 3
                elif "summary" in full:
                    new_phase = 4
                if new_phase > phase:
                    phase = new_phase
                    yield "phase", {"phase": phase}

            if choices[0].get("finish_reason") == "stop":
                break

    # Parse JSON result
    yield "phase", {"phase": 5}
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', full)
        if json_match:
            parsed = json.loads(json_match.group(0))
        else:
            # Fallback: extract structured info from free text
            parsed = _extract_from_text(full)
    except json.JSONDecodeError:
        parsed = _extract_from_text(full)

    parsed["raw_text"] = full
    yield "parsed", parsed


def _extract_from_text(text: str) -> dict:
    """Fallback structured extraction from markdown text."""
    sections = []
    current = {"heading": "分析结果", "content": ""}
    for line in text.split('\n'):
        if line.startswith('## ') or line.startswith('### '):
            if current["content"].strip():
                sections.append(current)
            current = {"heading": line.lstrip('#').strip(), "content": ""}
        else:
            current["content"] += line + '\n'
    if current["content"].strip():
        sections.append(current)

    entities = []
    # Simple entity extraction
    for match in re.finditer(r'\*\*(.+?)\*\*', text):
        name = match.group(1)
        if len(name) < 20 and name not in [e["name"] for e in entities]:
            entities.append({"name": name, "type": "概念", "weight": 5})

    return {
        "title": sections[0]["heading"] if sections else "分析结果",
        "entities": entities[:10] if entities else [
            {"name": "主题概念", "type": "概念", "weight": 5}
        ],
        "relations": [],
        "sections": sections if sections else [{"heading": "分析结果", "content": text}],
        "summary": text[:200] if text else "请查看详细分析结果",
        "raw_text": text,
    }


# ---------- HTTP Handler ----------
class ThreadedHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stdout.write(f"[{self.address_string()}] {args[0]}\n")
        sys.stdout.flush()

    def _json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _sse_init(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _sse(self, event: str, data):
        msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        self.wfile.write(msg.encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ("/", "/index.html"):
            html = HTML_FILE.read_text(encoding="utf-8")
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(data))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(data)
        elif path == "/api/history":
            self._json(200, {"history": get_history()})
        else:
            self.send_error(404)

    def do_POST(self):
        path = self.path.split('?')[0]
        if path == "/api/extract/stream":
            self._handle_stream()
        elif path == "/api/export":
            self._handle_export()
        else:
            self.send_error(404)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0 or length > 12 * 1024 * 1024:
            return None
        return json.loads(self.rfile.read(length))

    def _get_key(self):
        key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not key:
            self._json(400, {"error": "DASHSCOPE_API_KEY not set"})
            return None
        return key

    def _handle_stream(self):
        api_key = self._get_key()
        if not api_key:
            return
        body = self._read_body()
        if not body:
            self._json(400, {"error": "Invalid request"})
            return

        img = body.get("image", "")
        mime = body.get("mime", "image/png")
        if not img:
            self._json(400, {"error": "No image data"})
            return

        ck = cache_key(img)
        cached = cache_get(ck)
        if cached:
            self._sse_init()
            self._sse("cached", cached)
            self._sse("done", {"status": "complete"})
            return

        self._sse_init()
        self._sse("phase", {"phase": 0, "label": "正在上传分析..."})

        final = None
        try:
            for evt, data in stream_qwen_vl(api_key, img, mime):
                self._sse(evt, data)
                if evt == "parsed":
                    final = data
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            self._sse("error", {"error": f"AI 处理失败，正在重试... ({detail[:100]})"})
            return
        except Exception as e:
            self._sse("error", {"error": f"连接异常: {str(e)[:200]}"})
            return

        if final:
            final["id"] = uuid.uuid4().hex[:8]
            final["timestamp"] = int(time.time())
            final["mime"] = mime
            final["image"] = img[:200]  # thumbnail reference
            cache_set(ck, final)
            add_history(final)

        self._sse("done", {"status": "complete"})

    def _handle_export(self):
        body = self._read_body()
        if not body:
            self._json(400, {"error": "Invalid request"})
            return
        fmt = body.get("format", "json")
        data = body.get("data", {})
        if fmt == "markdown":
            md = f"# {data.get('title', 'Knowledge Extraction')}\n\n"
            for s in data.get("sections", []):
                md += f"## {s.get('heading', '')}\n\n{s.get('content', '')}\n\n"
            md += f"\n---\n**总结**: {data.get('summary', '')}"
            self._json(200, {"content": md, "type": "text/markdown"})
        else:
            self._json(200, {"content": json.dumps(data, ensure_ascii=False, indent=2), "type": "application/json"})


def main():
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not key:
        print("=" * 56)
        print("  提示: 请设置 DASHSCOPE_API_KEY 环境变量")
        print("=" * 56)
        print()

    addr = ("0.0.0.0", PORT)
    server = ThreadedHTTPServer(addr, Handler)
    url = f"http://localhost:{PORT}"
    print(f"\n  AI Knowledge Extractor v3")
    print(f"  {url}")
    print(f"  流式输出 | 知识图谱 | 历史记录 | 多格式导出\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()


if __name__ == "__main__":
    main()
