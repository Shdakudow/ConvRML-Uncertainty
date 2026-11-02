"""Durable local publisher: upload Exp 4 only after every result is verified.

Intended for a five-minute LaunchAgent. Work happens in isolated Git clones,
so an open demo checkout and other publishers' edits are never staged here.
"""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
WORK = Path.home() / ".local/share/convrml-exp4-20261101"
REMOTE = "/home/jy2228/ConvRML/experiments/experiment4_generalized_20261101"
REPOSITORY = "https://github.com/Shdakudow/ConvRML-Uncertainty-Demo.git"
ASSET_BRANCH = "experiment4-assets-20261101"
HOST = "jy2228@unicorn-login-02.coecis.cornell.edu"
SSH = [
    "ssh",
    "-i",
    str(Path.home() / ".ssh/id_ed25519"),
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    HOST,
]
ARMS = ("qutcc_clean50", "qutcc_general100", "im2im_clean50", "im2im_general100")


def run(args, cwd=None, **kwargs):
    return subprocess.check_output(
        [str(a) for a in args], cwd=cwd, text=True, **kwargs
    ).strip()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def status(stage, **values):
    WORK.mkdir(parents=True, exist_ok=True)
    row = dict(stage=stage, time=dt.datetime.now(dt.timezone.utc).isoformat(), **values)
    write(WORK / "status.json", row)
    print(json.dumps(row), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(folder):
    """Reject missing, changed, out-of-tree or scientifically incomplete exports."""
    ready = json.loads((folder / "READY.json").read_text())
    if ready.get("schema") != "experiment4-v1" or ready.get("passed") is not True:
        raise ValueError("Export is not complete")
    listed = set(ready["files"])
    actual = {
        str(p.relative_to(folder))
        for p in folder.rglob("*")
        if p.is_file() and p.name != "READY.json" and ".git" not in p.parts
    }
    if listed != actual or sum(p.endswith(".png") for p in listed) != 5000:
        raise ValueError("Unexpected or incomplete export file inventory")
    from PIL import Image

    for name, info in ready["files"].items():
        path = folder / name
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Out-of-tree export path")
        if path.stat().st_size != info["bytes"] or sha(path) != info["sha256"]:
            raise ValueError(f"Export hash mismatch: {name}")
        if name.endswith(".png"):
            with Image.open(path) as image:
                if image.size != (480, 270) or image.mode != "RGB":
                    raise ValueError("Wrong exported image shape/mode")
                image.verify()
    data = json.loads((folder / "data.json").read_text())
    if (
        data["schema"] != "experiment4-v1"
        or len(data["ids"]) != 50
        or len(set(data["ids"])) != 50
    ):
        raise ValueError("Wrong example cohort")
    if len(data["conditions"]) != 11 or set(data["arms"]) != set(ARMS):
        raise ValueError("Incomplete comparison")
    expected_images = {
        f"images/shared/{index:03d}-{kind}.png"
        for index in range(50)
        for kind in ("original", "target")
    }
    for condition in data["conditions"]:
        name = condition["id"]
        for index in range(50):
            if name != "clean":
                expected_images.add(f"images/{name}/{index:03d}-input.png")
            for arm in ARMS:
                for kind in ("prediction", "width"):
                    expected_images.add(f"images/{name}/{arm}/{index:03d}-{kind}.png")
    if {name for name in listed if name.endswith(".png")} != expected_images:
        raise ValueError("Display image paths do not match the comparison")
    for arm in ARMS:
        if data["arms"][arm]["epochs"] != (50 if arm.endswith("clean50") else 100):
            raise ValueError("Wrong training budget")
        fit = data["calibration"][arm]
        if fit["n_images"] != 3000 or fit["test_used_for_calibration"] is not False:
            raise ValueError("Wrong calibration cohort")
    for condition in data["conditions"]:
        row = data["results"][condition["id"]]
        report = json.loads(
            (folder / "reports" / (condition["id"] + ".json")).read_text()
        )
        if report["count"] != 2000 or report["shared_inputs_across_arms"] is not True:
            raise ValueError("Invalid full test report")
        if row["count"] != 2000 or set(row["arms"]) != set(ARMS):
            raise ValueError("Incomplete test comparison")
        for arm in ARMS:
            if len(row["arms"][arm]["examples"]) != 50:
                raise ValueError("Incomplete displayed metrics")
            full = report["arms"][arm]
            if (
                len(full["per_image"]) != 2000
                or full["examples"] != row["arms"][arm]["examples"]
                or full["mean"] != row["arms"][arm]["mean"]
            ):
                raise ValueError("Full test report disagrees with the page")
    return data


def asset_commit(folder):
    if not (folder / ".git").exists():
        run(["git", "init", "-b", ASSET_BRANCH], folder)
        run(["git", "remote", "add", "origin", REPOSITORY], folder)
    run(["git", "add", "--", "."], folder)
    if run(["git", "diff", "--cached", "--name-only"], folder):
        run(
            [
                "git",
                "commit",
                "-m",
                "Add verified Exp 4 calibration and paired reconstruction results",
            ],
            folder,
        )
    run(["git", "push", "origin", ASSET_BRANCH], folder, timeout=1800)
    return run(["git", "rev-parse", "HEAD"], folder)


def navigation(path):
    text = path.read_text()
    if "exp4:'experiment4/'" not in text:
        needle = "const routes={"
        if text.count(needle) != 1:
            raise ValueError("Shared navigation changed; manual integration needed")
        text = text.replace(needle, needle + "exp4:'experiment4/',", 1)
    if "nav.querySelector('[data-demo-route=\"exp4\"]')" not in text:
        needle = "  const state=new URLSearchParams(location.search);"
        if text.count(needle) != 1:
            raise ValueError("Shared navigation insertion point changed")
        addition = "  if(nav&&!nav.querySelector('[data-demo-route=\"exp4\"]')){\n   const link=document.createElement('a');link.dataset.demoRoute='exp4';link.textContent='4 · Generalized inputs';nav.append(link);\n  }\n"
        text = text.replace(needle, addition + needle, 1)
    path.write_text(text)


def publish_main(data, commit):
    checkout = WORK / "demo-main"
    if not (checkout / ".git").exists():
        run(
            [
                "git",
                "clone",
                "--single-branch",
                "--branch",
                "main",
                REPOSITORY,
                checkout,
            ],
            timeout=300,
        )
    run(["git", "fetch", "origin", "main"], checkout, timeout=120)
    if run(["git", "status", "--porcelain"], checkout):
        raise ValueError(
            "Isolated publisher checkout has uncommitted changes; publication deferred"
        )
    run(["git", "rebase", "origin/main"], checkout, timeout=120)
    destination = checkout / "experiment4"
    destination.mkdir(exist_ok=True)
    for name in ("index.html", "style.css", "app.js"):
        shutil.copy2(HERE / "web" / name, destination / name)
    data = dict(
        data,
        base=f"https://raw.githubusercontent.com/Shdakudow/ConvRML-Uncertainty-Demo/{commit}/",
    )
    write(destination / "data.json", data)
    navigation(checkout / "demo-navigation.js")
    run(["node", "--check", destination / "app.js"], checkout)
    run(["node", "--check", checkout / "demo-navigation.js"], checkout)
    run(["git", "diff", "--check"], checkout)
    run(["git", "add", "--", "experiment4", "demo-navigation.js"], checkout)
    names = run(["git", "diff", "--cached", "--name-only"], checkout).splitlines()
    if any(
        not (name.startswith("experiment4/") or name == "demo-navigation.js")
        for name in names
    ):
        raise ValueError("Unexpected staged files")
    if names:
        run(
            [
                "git",
                "commit",
                "-m",
                "Publish Exp 4: clean 50 versus generalized 100 with optional uncertainty",
            ],
            checkout,
        )
    run(["git", "push", "origin", "main"], checkout, timeout=300)
    return run(["git", "rev-parse", "HEAD"], checkout)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (WORK / "publisher.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    if (WORK / "published.json").exists():
        return
    expiry = WORK / "expires.json"
    if not expiry.exists():
        write(
            expiry,
            dict(
                expires=(
                    dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=7)
                ).isoformat()
            ),
        )
    if dt.datetime.now(dt.timezone.utc) > dt.datetime.fromisoformat(
        json.loads(expiry.read_text())["expires"]
    ):
        status("expired")
        return
    ready = run(
        SSH
        + [
            f"test -f {REMOTE}/output/web_data/READY.json && echo ready || echo waiting"
        ],
        timeout=40,
    )
    if ready != "ready":
        status("waiting_for_completed_calibration_and_test_export")
        return
    folder = WORK / "assets"
    folder.mkdir(exist_ok=True)
    status("downloading")
    run(
        [
            "rsync",
            "-az",
            "-e",
            "ssh -i " + str(Path.home() / ".ssh/id_ed25519") + " -o BatchMode=yes",
            f"{HOST}:{REMOTE}/output/web_data/",
            str(folder) + "/",
        ],
        timeout=1800,
    )
    status("verifying")
    data = verify(folder)
    status("checking_completed_page")
    browser_python = Path.home() / "Downloads/ConvRML_MRI_Local/.venv/bin/python"
    qa = run(
        [browser_python, HERE / "verify_web.py", "--data", folder / "data.json"],
        timeout=180,
    )
    (WORK / "browser-check.txt").write_text(qa + "\n")
    commit = asset_commit(folder)
    write(WORK / "uploaded-assets.json", dict(commit=commit))
    status("publishing_page", asset_commit=commit)
    page = publish_main(data, commit)
    url = "https://shdakudow.github.io/ConvRML-Uncertainty-Demo/experiment4/"
    write(WORK / "published.json", dict(commit=page, asset_commit=commit, url=url))
    status("published", commit=page, asset_commit=commit, url=url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        verify(args.verify_only)
    else:
        try:
            main()
        except Exception as error:
            status("retry_after_error", error=repr(error))
            raise
