#!/usr/bin/env python3
import http.server
import socketserver
from pathlib import Path

PORT = 8888
ROOT = Path(__file__).parent.resolve()

# Windows often maps .js to text/plain; module scripts need a JS MIME type.
_MIME = http.server.SimpleHTTPRequestHandler.extensions_map.copy()
_MIME.update({".js": "text/javascript", ".mjs": "text/javascript"})


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = _MIME

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        # Quiet missing /favicon.ico
        if self.path.split("?")[0] in ("/favicon.ico", "favicon.ico") and not (ROOT / "favicon.ico").is_file():
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving {ROOT} at http://127.0.0.1:{PORT}/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
