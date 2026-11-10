"""Live stage display of a scan: the Dino's video and the tower with the block being scanned lit.

A small web server (standard library only, 127.0.0.1) inside the scan process:

    /         the page (dashboard.html), full screen in any browser: F = full screen, L = language
    /stream   the Dino as MJPEG
    /events   the scan state as server-sent events

    uv run jenga-autoscan run --send --dashboard       # with the real scan
    uv run jenga-autoscan dashboard captures/autoscan_A captures/autoscan_B     # replay, no robot or camera

The replay plays saved sessions: focus_log.csv gives the order and the focus values, the saved shot
(blurred according to the focus value) stands in for the video.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

PAGE = Path(__file__).with_name("dashboard.html")
DEFAULT_PORT = 8765
JPEG_QUALITY = 85
MAX_FPS = 20.0
REPLAY_MM_PER_PX = 0.0156  # dial 40, 2026-11-03


class NoDashboard:
    """Used when there is no dashboard: every call does nothing."""

    def event(self, **_) -> None:
        pass

    def shot(self, **_) -> None:
        pass

    def frame(self, _) -> None:
        pass


class Dashboard(NoDashboard):
    def __init__(self, port: int = DEFAULT_PORT):
        self.state: dict = {"stage": "idle", "scanned": [], "order": [], "total": 0, "focus": None, "peak": None,
                            "started": None, "ended": None}
        self.version = 0
        self.changed = threading.Condition()
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.new_frame = threading.Condition()
        self.last_frame_t = 0.0
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def do_GET(self):
                try:
                    if self.path.split("?")[0] == "/":
                        self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")
                    elif self.path == "/state":
                        self.reply(json.dumps(dashboard.state).encode(), "application/json")
                    elif self.path == "/events":
                        self.events()
                    elif self.path == "/stream":
                        self.stream()
                    else:
                        self.send_error(404)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the browser went away

            def reply(self, body: bytes, kind: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def events(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()
                seen = -1
                while True:
                    with dashboard.changed:
                        dashboard.changed.wait_for(lambda: dashboard.version != seen, timeout=10)
                        seen = dashboard.version
                        data = json.dumps(dashboard.state)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()

            def stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()
                seen = -1
                while True:
                    with dashboard.new_frame:
                        dashboard.new_frame.wait_for(lambda: dashboard.frame_id != seen, timeout=10)
                        seen, jpeg = dashboard.frame_id, dashboard.jpeg
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                    self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{port}/"

    def event(self, **fields) -> None:
        """Merge into the state; the page redraws from the whole state."""
        with self.changed:
            if fields.get("stage") == "scan" and self.state["started"] is None:
                self.state["started"] = time.time()
            self.state.update(fields)
            self.version += 1
            self.changed.notify_all()

    def shot(self, **block) -> None:
        with self.changed:
            self.state["scanned"] = [b for b in self.state["scanned"]
                                     if (b["layer"], b["slot"]) != (block["layer"], block["slot"])] + [block]
            self.version += 1
            self.changed.notify_all()

    def frame(self, frame: np.ndarray) -> None:
        now = time.time()
        if now - self.last_frame_t < 1 / MAX_FPS:
            return
        self.last_frame_t = now
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with self.new_frame:
                self.jpeg = jpeg.tobytes()
                self.frame_id += 1
                self.new_frame.notify_all()


# --- replay of saved sessions ---------------------------------------------------------
def blurred(image: np.ndarray, focus: float, peak: float) -> np.ndarray:
    """What the Dino saw at this focus value, near enough for a rehearsal."""
    sigma = 14.0 * (1.0 - min(focus / max(peak, 1e-6), 1.0))
    return image if sigma < 0.3 else cv2.GaussianBlur(image, (0, 0), sigma)


def replay(dash: Dashboard, sessions: list[Path], speed: float, order: list[dict]) -> None:
    def wait(seconds: float) -> None:
        time.sleep(seconds / speed)

    runs = []
    for session in sessions:
        with (session / "shots.csv").open() as f:
            shots = list(csv.DictReader(f))
        with (session / "focus_log.csv").open() as f:
            log = list(csv.DictReader(f))
        runs.append((session, shots, log))
    dash.event(stage="idle", total=sum(len(s) for _, s, _ in runs), scanned=[], order=[], started=None, ended=None)
    wait(3)
    dash.event(stage="pick")
    wait(20)
    for session, shots, log in runs:
        face = shots[0]["face"]
        by_block = {(r["layer"], r["slot"]): r for r in shots}
        for n, row in enumerate(log):
            shot = by_block.get((row["layer"], row["slot"]))
            if shot is None:
                continue
            image = cv2.imread(str(session / shot["image"]))
            peak = float(shot["focus"]) or 1.0
            value = float(row["focus"])
            dash.frame(blurred(image, value, max(peak, value)))
            dash.event(stage="scan", face=face, layer=int(row["layer"]), slot=int(row["slot"]), action="focus",
                       focus=value, peak=peak)
            wait(1.3)
            next_block = log[n + 1] if n + 1 < len(log) else None
            if next_block is None or (next_block["layer"], next_block["slot"]) != (row["layer"], row["slot"]):
                dash.frame(image)
                dash.event(action="edge", focus=peak)
                wait(3)
                gap = float(shot["gap"])
                dash.event(action="shot")
                wait(1.5)
                dash.shot(layer=int(row["layer"]), slot=int(row["slot"]), face=face, gap_px=gap,
                          gap_mm=round(gap * REPLAY_MM_PER_PX, 3))
                dash.event(action="measured", gap_mm=round(gap * REPLAY_MM_PER_PX, 3))
                wait(3)
    dash.event(stage="place", order=order)
    wait(20)
    dash.event(stage="done", ended=time.time())
