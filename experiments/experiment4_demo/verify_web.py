"""Offline UI check in standalone Chrome, using clearly marked synthetic fixtures.

Used only when the in-app browser connection is unavailable. No fixture results
are written to the demo or uploaded. Requires websocket-client and Pillow in the
local QA environment. Optionally --data /path/to/completed/data.json checks actual
result structure with the same local image fixture to avoid network dependencies.
"""

import argparse
import ast
import base64
import functools
import http.server
from io import BytesIO
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request

from PIL import Image, ImageDraw
import websocket

HERE = Path(__file__).resolve().parent
ARMS = ("qutcc_clean50", "qutcc_general100", "im2im_clean50", "im2im_general100")


def fixture():
    tree = ast.parse((HERE / "conditions.py").read_text())
    rows = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CONDITIONS" for t in node.targets)
    )
    metrics = dict(
        psnr=25.0, ssim=0.8, mae=0.03, coverage=0.9, width=0.12, crossing=0.0
    )
    return dict(
        schema="experiment4-v1",
        base="./",
        ids=[f"UI FIXTURE {i+1}" for i in range(50)],
        conditions=[
            dict(id=i, title=t, description=d, formula=f) for i, t, d, f in rows
        ],
        results={
            r[0]: dict(
                count=2000,
                arms={
                    a: dict(
                        mean=metrics,
                        examples=[
                            dict(metrics, ssim=0.8 - i * 0.001) for i in range(50)
                        ],
                    )
                    for a in ARMS
                },
            )
            for r in rows
        },
        calibration={
            a: dict(status="fitted", target_met=True, calibration_coverage=0.9)
            for a in ARMS
        },
    )


def main(data_path=None):
    data = json.loads(data_path.read_text()) if data_path else fixture()
    data["base"] = "./"
    image = Image.new("RGB", (480, 270), "#183047")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 40, 420, 230), fill="#239ba3")
    draw.text((110, 130), "LOCAL UI FIXTURE - NOT A RESULT", fill="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    png = buffer.getvalue()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path.endswith("/data.json"):
                body = json.dumps(data).encode()
                content = "application/json"
            elif path.endswith(".png"):
                body = png
                content = "image/png"
            elif path == "/demo-navigation.js":
                body = b""
                content = "text/javascript"
            else:
                return super().do_GET()
            self.send_response(200)
            self.send_header("Content-Type", content)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(HERE / "web"))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    profile = tempfile.mkdtemp(prefix="convrml-exp4-browser-")
    log = (Path(tempfile.gettempdir()) / "convrml-exp4-chrome.log").open("w")
    process = subprocess.Popen(
        [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--disable-background-networking",
            f"--remote-debugging-port={port}",
            f"--remote-allow-origins=http://localhost:{port}",
            "--user-data-dir=" + profile,
            "about:blank",
        ],
        stdout=log,
        stderr=log,
    )
    ws = None
    try:
        for _ in range(100):
            try:
                tabs = json.load(
                    urllib.request.urlopen(f"http://localhost:{port}/json", timeout=1)
                )
                tab = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("Chrome debugging connection unavailable")
        ws = websocket.create_connection(
            tab["webSocketDebuggerUrl"], origin=f"http://localhost:{port}", timeout=30
        )
        seq = 0
        errors = []

        def call(method, params=None):
            nonlocal seq
            seq += 1
            ws.send(json.dumps(dict(id=seq, method=method, params=params or {})))
            while True:
                value = json.loads(ws.recv())
                if value.get("method") == "Runtime.exceptionThrown":
                    errors.append(value)
                if value.get("id") == seq:
                    if "error" in value:
                        raise RuntimeError(value)
                    return value.get("result", {})

        def js(code):
            value = call(
                "Runtime.evaluate",
                dict(expression=code, returnByValue=True, awaitPromise=True),
            )
            if "exceptionDetails" in value:
                raise RuntimeError(value)
            return value.get("result", {}).get("value")

        call("Runtime.enable")
        call(
            "Emulation.setDeviceMetricsOverride",
            dict(width=1440, height=1100, deviceScaleFactor=1, mobile=False),
        )
        call(
            "Page.navigate",
            dict(url=f"http://127.0.0.1:{server.server_address[1]}/?example=0"),
        )
        for _ in range(100):
            if js("document.querySelector('#experiment')?.hidden === false"):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(js("document.querySelector('#status')?.textContent"))
        assert js("document.querySelectorAll('.condition').length") == 11
        assert js("document.querySelectorAll('details[open]').length") == 0
        assert js("document.querySelectorAll('img[data-kind=width]').length") == 0
        assert js("document.querySelectorAll('#scene option').length") == 50
        assert (
            js(
                "document.querySelectorAll('.condition:first-child img[data-kind=prediction]').length"
            )
            == 4
        )
        js("document.querySelector('.condition summary').click()")
        time.sleep(0.15)
        assert js("document.querySelectorAll('details[open]').length") == 1
        assert (
            js(
                "document.querySelectorAll('.condition:first-child img[data-kind=width]').length"
            )
            == 4
        )
        assert (
            js("document.querySelectorAll('.condition:first-child table').length") == 2
        )
        assert js(
            "document.querySelector('.condition:first-child table').textContent.includes('QUTCC · generalized · 100 epochs')"
        )
        old = js(
            "document.querySelector('.condition:first-child img[data-kind=prediction]').src"
        )
        js("document.querySelector('#next').click()")
        assert js("document.querySelector('#scene').value") == "1"
        assert js("new URL(location.href).searchParams.get('example')") == "1"
        assert (
            js(
                "document.querySelector('.condition:first-child img[data-kind=prediction]').src"
            )
            != old
        )
        js(
            "document.querySelector('#scene').value='49';document.querySelector('#scene').dispatchEvent(new Event('change'))"
        )
        assert js("document.querySelector('#next').disabled")
        js("document.querySelector('#previous').click()")
        assert js("document.querySelector('#scene').value") == "48"
        # Capture a clean first-scene view, then mobile and expanded detail views.
        js(
            "document.querySelector('#scene').value='0';document.querySelector('#scene').dispatchEvent(new Event('change'));document.querySelector('details').open=false;window.scrollTo(0,0)"
        )
        for name, width, height in [("desktop", 1440, 1100), ("mobile", 390, 844)]:
            call(
                "Emulation.setDeviceMetricsOverride",
                dict(
                    width=width, height=height, deviceScaleFactor=1, mobile=width < 500
                ),
            )
            time.sleep(0.25)
            assert js("document.documentElement.scrollWidth <= window.innerWidth + 1")
            shot = call("Page.captureScreenshot", dict(format="png", fromSurface=True))
            (Path(tempfile.gettempdir()) / f"convrml-exp4-{name}.png").write_bytes(
                base64.b64decode(shot["data"])
            )
        call(
            "Emulation.setDeviceMetricsOverride",
            dict(width=1440, height=1100, deviceScaleFactor=1, mobile=False),
        )
        js(
            "document.querySelector('details').open=true;document.querySelector('details').scrollIntoView()"
        )
        time.sleep(0.25)
        shot = call("Page.captureScreenshot", dict(format="png", fromSurface=True))
        (Path(tempfile.gettempdir()) / "convrml-exp4-details.png").write_bytes(
            base64.b64decode(shot["data"])
        )
        assert not errors, errors
        print(
            "PASS: 11 sections; four reconstructions; uncertainty hidden initially; expand/collapse; scene changes; matching URLs; last-scene controls; desktop/mobile overflow; no JavaScript errors."
        )
    finally:
        if ws:
            ws.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        server.shutdown()
        server.server_close()
        log.close()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    main(parser.parse_args().data)
