"""
图片知识点总结器 - 本地服务器
支持 Google Gemini（免费）和 Anthropic Claude
使用前设置环境变量: set GEMINI_API_KEY=你的key
然后运行: python server.py
"""

import http.server
import json
import os
import re
import sys
import urllib.request
import webbrowser
from pathlib import Path

PORT = int(os.environ.get("PORT", 8765))
HTML_FILE = Path(__file__).parent / "image-knowledge-summary.html"

PROMPT = (
    "请仔细分析这张图片，提取所有关键知识点，并用结构化、清晰的方式总结。\n"
    "要求：\n"
    "1. 用中文输出\n"
    "2. 按主题或类别分组，使用标题层级\n"
    "3. 对每个知识点给出简洁但完整的解释\n"
    "4. 如果图片包含公式、代码或数据，请精确还原\n"
    "5. 用列表、表格等方式增强可读性\n"
    "6. 最后给出一个简短的总结，说明图片的核心内容"
)

# ---------- Markdown to HTML renderer ----------

def render_markdown(md: str) -> str:
    html = md

    html = re.sub(r'```(\w*)\n(.*?)```', lambda m:
        f'<pre><code>{_escape(m.group(2).rstrip())}</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^[\-\*] (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'((?:<li>.*?</li>\n?)+)', r'<ul>\1</ul>', html)
    html = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
    html = re.sub(r'^---+$', r'<hr>', html, flags=re.MULTILINE)
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

def _escape(s: str) -> str:
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

# ---------- Gemini API ----------

def call_gemini(api_key: str, image_data: str, mime_type: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    body = json.dumps({
        "contents": [{
            "parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": image_data}},
            ]
        }]
    })

    req = urllib.request.Request(
        url, data=body.encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["candidates"][0]["content"]["parts"][0]["text"]

# ---------- Anthropic API (fallback) ----------

def call_anthropic(api_key: str, image_data: str, mime_type: str) -> str:
    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 2048,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime_type, "data": image_data},
                },
                {"type": "text", "text": PROMPT},
            ],
        }],
    })
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["content"][0]["text"]

# ---------- HTTP server ----------

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stdout.write(f"[{self.address_string()}] {args[0]}\n")
        sys.stdout.flush()

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = HTML_FILE.read_text(encoding="utf-8")
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/summarize":
            # Prefer Gemini (free), fall back to Anthropic
            gemini_key = os.environ.get("GEMINI_API_KEY", "")
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

            if not gemini_key and not anthropic_key:
                self._send_json(400, {"error": "服务器未配置 API Key"})
                return

            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                self._send_json(400, {"error": "请求体为空"})
                return
            if length > 12 * 1024 * 1024:
                self._send_json(400, {"error": "图片太大，限制 10MB"})
                return

            body = json.loads(self.rfile.read(length))
            image_data = body.get("image")
            mime_type = body.get("mime", "image/png")

            if not image_data:
                self._send_json(400, {"error": "未提供图片数据"})
                return

            try:
                if gemini_key:
                    text = call_gemini(gemini_key, image_data, mime_type)
                else:
                    text = call_anthropic(anthropic_key, image_data, mime_type)
                html = render_markdown(text)
                self._send_json(200, {"html": html, "text": text})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        else:
            self.send_error(404)

def main():
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not gemini_key and not anthropic_key:
        print("=" * 56)
        print("  提示: 请设置 API Key 环境变量")
        print("  Gemini (免费):  set GEMINI_API_KEY=你的key")
        print("  Anthropic:      set ANTHROPIC_API_KEY=sk-ant-...")
        print("=" * 56)
        print()

    addr = ("0.0.0.0", PORT)
    server = http.server.HTTPServer(addr, Handler)
    url = f"http://localhost:{PORT}"
    print(f"服务已启动: {url}")
    print("按 Ctrl+C 停止服务\n")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()

if __name__ == "__main__":
    main()
