"""Restore one baseline epoch at a time and evaluate without retraining.

Uses the verified GitHub Release manifests already created for this project.
The paper-model array uses one GPU; baseline evaluations use up to six public GPUs.
Temporary restored weights are removed only after every task in their round has
a result.
"""

import concurrent.futures
import datetime
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.error
import urllib.request

from coverage90_cluster import validate_manifest

WORK = Path.home() / ".local/share/convrml-coverage90-20261022"
ARCHIVE = Path.home() / ".local/share/convrml-checkpoint-archive"
REMOTE = "/home/jy2228/ConvRML/experiments/coverage90_20261022"
HOST = "jy2228@unicorn-login-02.coecis.cornell.edu"
SSH = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-i",
    str(Path.home() / ".ssh/id_ed25519"),
    "-o",
    "ConnectTimeout=20",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
    HOST,
]
ARMS = {"im2im", "mc_005", *(f"ensemble_{i:02d}" for i in range(5))}


def run(command, **kwargs):
    return subprocess.check_output(
        [str(v) for v in command], text=True, **kwargs
    ).strip()


def remote(action, payload, timeout=300):
    """Send structured input instead of constructing Python source inside SSH."""
    command = shlex.join(
        [
            "/home/jy2228/.conda/envs/convrml/bin/python",
            REMOTE + "/coverage90_cluster.py",
            action,
        ]
    )
    return json.loads(run(SSH + [command], input=json.dumps(payload), timeout=timeout))


def entries(epoch):
    result = []
    for camera in ("rml", "diffuser"):
        manifest = json.loads((ARCHIVE / (camera + "-manifest.json")).read_text())
        for entry in manifest["files"]:
            p = Path(entry["path"])
            if (
                len(p.parts) == 4
                and p.parts[1] in ARMS
                and p.name == f"epoch_{epoch:03d}.pt"
            ):
                result.append(dict(entry, release=manifest["release"]))
    validate_manifest(epoch, result)
    return result


def inventory(epoch, required):
    return remote("inventory", dict(epoch=epoch, entries=required))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def signed_entries(required):
    """Authorize individual downloads locally; never send GitHub credentials to SSH."""
    token = run(["gh", "auth", "token"])
    assets = {}
    for release in {entry["release"] for entry in required}:
        metadata = json.loads(run(
            ["gh", "api", f"repos/Shdakudow/ConvRML-Uncertainty/releases/tags/{release}"],
            timeout=60,
        ))
        assets[release] = {asset["name"]: asset for asset in metadata["assets"]}

    def sign(entry):
        asset = assets[entry["release"]][entry["name"]]
        if asset["size"] != entry["size"]:
            raise ValueError("Release asset size differs from archived manifest")
        request = urllib.request.Request(
            asset["url"] + "?download=" + str(time.time_ns()),
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/octet-stream",
            },
        )
        try:
            urllib.request.build_opener(NoRedirect).open(request, timeout=30).close()
        except urllib.error.HTTPError as error:
            if error.code in (301, 302, 303, 307, 308):
                return dict(entry, download_url=error.headers["Location"])
            raise RuntimeError(
                f"Download authorization failed: HTTP {error.code}"
            ) from None
        raise RuntimeError("Expected a temporary GitHub Release download URL")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(sign, required))


def stage(epoch, required):
    if inventory(epoch, required)["missing"]:
        # Only small, temporary single-file URLs cross SSH, never checkpoint bytes.
        result = remote(
            "restore", dict(epoch=epoch, entries=signed_entries(required)), timeout=1800
        )
        print("Direct checkpoint restoration", json.dumps(result), flush=True)


def submit(epoch, tasks, kind="baseline"):
    parallelism = 6 if kind == "baseline" else 1
    response = remote(
        "submit", dict(epoch=epoch, tasks=tasks, kind=kind, parallelism=parallelism)
    )
    return response["job"]


def status(job, tasks):
    return remote("status", dict(job=job, tasks=tasks))


def cleanup(epoch, required, tasks):
    """Only discard this pipeline's copies after every member has a valid result."""
    remote("cleanup", dict(epoch=epoch, entries=required, tasks=tasks))
    for entry in required:
        target = WORK / "milestone_cache" / f"epoch_{epoch:03d}" / entry["path"]
        target.unlink(missing_ok=True)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (WORK / "milestone_dispatch.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    path = WORK / "milestone_dispatch.json"
    state = json.loads(path.read_text()) if path.exists() else dict(rounds={})
    tasks = json.loads((WORK / "milestone_tasks.json").read_text())

    def save():
        state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(path)

    try:
        paper_tasks = [
            task for task in tasks if "--weights-root" not in task["command"]
        ]
        paper = state.setdefault(
            "paper",
            dict(
                job=(WORK / "paper_milestone_job.txt").read_text().strip(), attempts=1
            ),
        )
        paper["progress"] = status(paper["job"], paper_tasks)
        if paper["progress"]["active"]:
            paper["status"] = (
                "running"
                if any(job["state"] == "RUNNING" for job in paper["progress"]["active"])
                else "queued"
            )
        elif not paper["progress"]["missing"]:
            paper["status"] = "complete"
        elif paper["attempts"] < 3:
            remaining = [
                task
                for task in paper_tasks
                if task["output"] in paper["progress"]["missing"]
            ]
            paper.update(
                job=submit(0, remaining, "paper-retry"),
                attempts=paper["attempts"] + 1,
                status="submitted",
            )
        else:
            paper["status"] = "needs_attention"
        save()
        for epoch in range(5, 50, 5):
            round_tasks = [
                task
                for task in tasks
                if task["epoch"] == epoch and "--weights-root" in task["command"]
            ]
            to_submit = round_tasks
            row = state["rounds"].setdefault(
                str(epoch), dict(status="pending", attempts=0)
            )
            if row["status"] == "complete":
                continue
            required = entries(epoch)
            if row.get("job"):
                progress = status(row["job"], round_tasks)
                row["progress"] = progress
                if progress["active"]:
                    row["status"] = (
                        "running"
                        if any(job["state"] == "RUNNING" for job in progress["active"])
                        else "queued"
                    )
                    save()
                    return
                if not progress["missing"]:
                    cleanup(epoch, required, round_tasks)
                    row["status"] = "complete"
                    save()
                    continue
                if row["attempts"] >= 3:
                    row["status"] = "needs_attention"
                    save()
                    return
                to_submit = [
                    task
                    for task in round_tasks
                    if task["output"] in progress["missing"]
                ]
            row["status"] = "restoring_verified_checkpoints"
            save()
            stage(epoch, required)
            job = submit(epoch, to_submit)
            if job is None:
                cleanup(epoch, required, round_tasks)
                row["status"] = "complete"
                save()
                continue
            row.update(job=job, status="submitted", attempts=row["attempts"] + 1)
            state.pop("last_error", None)
            save()
            print("Submitted baseline epoch", epoch, "job", row["job"], flush=True)
            return
        state["complete"] = paper["status"] == "complete"
        save()
    except Exception as exc:
        state["last_error"] = repr(exc)
        save()
        raise


if __name__ == "__main__":
    main()
