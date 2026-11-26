"""Local web server for the review interface (Python standard library only)."""
from __future__ import annotations

import json
import logging
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .session import ReviewSession
from .uploads import MAX_UPLOAD, AnnotationDesk, save_input_image, start_analyse_new
from . import keybindings

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"


def make_handler(session: ReviewSession, keys_path=None):
    desk = AnnotationDesk(session)
    def _job():
        return {k: session.job.get(k) for k in ("kind", "state", "done", "total", "message", "result")}

    class Handler(BaseHTTPRequestHandler):
        server_version = "SeedReview/1.0"

        def log_message(self, fmt, *args):  # keep the console quiet
            log.debug(fmt, *args)

        # -------------------------------------------------------- helpers
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code: int = 200):
            self._send(code, json.dumps(data, default=float).encode("utf-8"), "application/json; charset=utf-8")

        def _error(self, msg: str, code: int = 400):
            self._json({"error": msg}, code)

        def _cfg(self, q):
            def num(name):
                v = q.get(name, [None])[0]
                return float(v) if v not in (None, "") else None
            return session.config_with(num("review"), num("reject"))

        def _image_id(self, m) -> int:
            i = int(m.group(1))
            if not 0 <= i < len(session.keys):
                raise IndexError
            return i

        def _raw(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                raise ValueError("file too large (max 300 MB)")
            data = b""
            while len(data) < n:
                chunk = self.rfile.read(min(1 << 20, n - len(data)))
                if not chunk:
                    break
                data += chunk
            return data

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        # -------------------------------------------------------- routes
        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/favicon.ico":
                    return self._send(204, b"", "image/x-icon")
                if u.path in ("/", "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/images":
                    cfg = self._cfg(q)
                    listing = session.listing(cfg)
                    return self._json({"images": listing, "totals": session.totals(listing),
                                       "output_dir": str(session.output_dir), "job": _job(),
                                       "thresholds": {"review": cfg["confidence"]["review_threshold"],
                                                      "reject": cfg["confidence"]["reject_threshold"]}})
                if u.path == "/api/keybindings":  # read on every page load, so edits apply after a reload
                    binds, warns = keybindings.load(keys_path)
                    return self._json({"bindings": binds, "descriptions": keybindings.descriptions(),
                                       "warnings": warns, "file": str(keys_path) if keys_path else None})
                if u.path == "/api/annotations":
                    return self._json(desk.listing())
                m = re.fullmatch(r"/api/annotations/preview/([0-9a-f]{32})", u.path)
                if m:
                    return self._send(200, desk.preview(m.group(1)), "image/jpeg")
                if u.path == "/api/job":
                    return self._json(_job())
                if u.path == "/api/removed":
                    return self._json({"removed": session.removed()})
                if u.path == "/api/settings":
                    return self._json(session.settings_report())
                if u.path == "/api/settings/file":
                    body = session.settings_file()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-yaml; charset=utf-8")
                    self.send_header("Content-Disposition", 'attachment; filename="seedcounter-settings.yaml"')
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                m = re.fullmatch(r"/api/image/(\d+)", u.path)
                if m:
                    return self._json(session.payload(self._image_id(m), self._cfg(q)))
                m = re.fullmatch(r"/api/image/(\d+)/picture", u.path)
                if m:
                    return self._send(200, session.picture(self._image_id(m)), "image/jpeg")
                return self._error("not found", 404)
            except IndexError:
                return self._error("no such image", 404)
            except Exception as exc:
                log.exception("GET %s failed", self.path)
                return self._error(f"server error: {exc}", 500)

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                name = q.get("name", [""])[0]
                if u.path == "/api/upload/image":
                    return self._json(save_input_image(session, name, self._raw()))
                if u.path == "/api/settings/load":
                    session.start_load_settings(self._raw())
                    return self._json(_job())
                if u.path == "/api/upload/annotation":
                    return self._json(desk.stage(q.get("token", [""])[0], q.get("role", [""])[0], name, self._raw()))
                body = self._body()
                cfg = self._cfg(q)
                if u.path == "/api/images/analyse":
                    start_analyse_new(session)
                    return self._json(_job())
                if u.path == "/api/annotations/check":
                    return self._json(desk.check(body.get("token", "")))
                if u.path == "/api/annotations/save":
                    return self._json(desk.save(body.get("token", ""), body.get("pod_counts")))
                if u.path == "/api/annotations/cancel":
                    desk.cancel(body.get("token", ""))
                    return self._json({"ok": True})
                if u.path == "/api/annotations/remove":
                    return self._json(desk.remove(str(body.get("image", ""))))
                m = re.fullmatch(r"/api/image/(\d+)/(edit|undo|discard)", u.path)
                if m:
                    i = self._image_id(m)
                    key = session.keys[i]
                    if m.group(2) == "edit":
                        session.apply(key, body)
                    elif m.group(2) == "undo":
                        session.undo(key)
                    else:
                        session.discard(key)
                    return self._json(session.payload(i, cfg))
                if u.path == "/api/export":
                    keys = body.get("images")
                    return self._json(session.export(cfg, None if keys is None else [str(k) for k in keys]))
                m = re.fullmatch(r"/api/image/(\d+)/remove", u.path)
                if m:
                    return self._json(session.remove(session.keys[self._image_id(m)]))
                if u.path == "/api/removed/restore":
                    return self._json(session.restore(str(body.get("key", ""))))
                if u.path == "/api/settings/reset":
                    session.start_reset_settings()
                    return self._json(_job())
                if u.path == "/api/calibrate":
                    session.start_calibration(bool(body.get("use_gt", True)), bool(body.get("sweep", True)))
                    return self._json(_job())
                if u.path == "/api/calibrate/apply":
                    session.start_apply_calibration()
                    return self._json(_job())
                return self._error("not found", 404)
            except IndexError:
                return self._error("no such image", 404)
            except (KeyError, ValueError, TypeError) as exc:
                return self._error(str(exc).strip("'\""))
            except Exception as exc:
                log.exception("POST %s failed", self.path)
                return self._error(f"server error: {exc}", 500)

    return Handler


def serve(session: ReviewSession, host: str = "127.0.0.1", port: int = 8000, keys_path=None) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(session, keys_path))
    httpd.daemon_threads = True
    return httpd
