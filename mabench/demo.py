"""Side-by-side viewer that follows run traces live or replays saved ones."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .sample import run_sample

PAGE = Path(__file__).with_name("demo.html")


def list_runs(root) -> list:
    root = Path(root)
    runs = [{"system": t.parts[-4], "budget": t.parts[-3], "instance": t.parts[-2]}
            for t in root.glob("*/*/*/trace.jsonl")]
    return sorted(runs, key=lambda r: (r["instance"], int(r["budget"]), r["system"]))


def read_trace(root, system: str, budget: str, instance: str) -> list:
    """Only runs found by listing are served, so request values can't reach other paths."""
    if {"system": system, "budget": budget, "instance": instance} not in list_runs(root):
        raise KeyError("unknown run")
    events = []
    for line in (Path(root) / system / budget / instance / "trace.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:  # the writer may be mid-line during a live run
            break
    return events


def handler(root):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/runs":
                self._send(200, json.dumps(list_runs(root)).encode())
            elif url.path == "/api/trace":
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                try:
                    body = read_trace(root, q["system"], q["budget"], q["instance"])
                    self._send(200, json.dumps(body).encode())
                except KeyError:
                    self._send(404, b'{"error":"unknown run"}')
            else:
                self._send(404, b'{"error":"not found"}')

        def log_message(self, *args):
            pass

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp", nargs="?", default="results/sample")
    p.add_argument("--sample", action="store_true", help="create scripted zero-cost runs first")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    if a.sample:
        run_sample(a.exp)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), handler(a.exp))
    print(f"demo at http://127.0.0.1:{a.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
