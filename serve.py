#!/usr/bin/env python3
"""Static server with caching disabled + dev telemetry (page POSTs state to /state)."""
import http.server
import pathlib
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8413
STATE = pathlib.Path('.dd_state.json')


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_POST(self):
        if self.path == '/state':
            n = int(self.headers.get('Content-Length', 0))
            STATE.write_bytes(self.rfile.read(n))
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


http.server.test(HandlerClass=Handler, port=PORT, bind='127.0.0.1')
