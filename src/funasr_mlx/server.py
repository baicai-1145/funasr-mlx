"""OpenAI-compatible local transcription server (stdlib-only HTTP).

POST /v1/audio/transcriptions  (multipart file= or raw body)
GET  /v1/models

The transcriber is a resident ``LlmAsrTranscriber`` (one model load, KV
reset-reuse between requests). Designed to run INSIDE the guarded child via
``funasr-mlx serve``; the parent process holds the memory policy.
"""
from __future__ import annotations

import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def run_server(checkpoint: Path, *, family: str, host: str, port: int, max_tokens: int) -> None:
    from .llm_asr import LlmAsrTranscriber

    transcriber = LlmAsrTranscriber(checkpoint, family=family, max_tokens=max_tokens)
    transcriber._ckpt_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [{"id": family, "object": "model", "owned_by": "funasr-mlx"}],
                })
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/v1/audio/transcriptions":
                self._json(404, {"error": "not found"})
                return
            ctype = self.headers.get("Content-Type", "")
            try:
                if ctype.startswith("multipart/form-data"):
                    audio_bytes, ext = self._parse_multipart(ctype)
                else:
                    length = int(self.headers.get("Content-Length", "0"))
                    audio_bytes = self.rfile.read(length)
                    ext = ".wav"
                if not audio_bytes:
                    self._json(400, {"error": "empty audio"})
                    return
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                try:
                    with lock:
                        result = transcriber.transcribe_file(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                self._json(200, {
                    "text": result.text,
                    "family": result.family,
                    "wall_ms": result.wall_ms,
                    "token_count": len(result.token_ids),
                })
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": str(exc)})

        def _parse_multipart(self, ctype: str) -> tuple[bytes, str]:
            import re

            m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', ctype)
            boundary = (m.group(1) or m.group(2)).encode()
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length)
            parts = data.split(b"--" + boundary)
            for part in parts:
                if b"filename=" not in part:
                    continue
                header_blob, _, payload = part.partition(b"\r\n\r\n")
                fm = re.search(rb'filename="([^"]+)"', header_blob)
                name = fm.group(1).decode() if fm else "audio.wav"
                payload = payload.rsplit(b"\r\n", 1)[0]
                return payload, ("." + name.rsplit(".", 1)[-1] if "." in name else ".wav")
            raise ValueError("multipart body has no file part")

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"funasr-mlx server: {family} on http://{host}:{port}/v1/audio/transcriptions", flush=True)
    httpd.serve_forever()
