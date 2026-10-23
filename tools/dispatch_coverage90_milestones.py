"""Restore one baseline epoch at a time and evaluate without retraining.

Uses the verified GitHub Release manifests already created for this project.
The paper-model array uses one GPU; each baseline array uses one more. Temporary
restored weights are removed only after every task in their round has a result.
"""

import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

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


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
                assert not p.is_absolute() and ".." not in p.parts
                result.append(dict(entry, release=manifest["release"]))
    validate_manifest(epoch, result)
    return result


def inventory(epoch, required):
    return remote("inventory", dict(epoch=epoch, entries=required))


def download(epoch, entry):
    folder = WORK / "milestone_cache" / f"epoch_{epoch:03d}"
    target = folder / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if (
        target.exists()
        and target.stat().st_size == entry["size"]
        and digest(target) == entry["sha256"]
    ):
        return
    local = (
        Path.home()
        / ".local/share/convrml-exp2-analysis-20261019/milestones"
        / f"epoch_{epoch:03d}"
        / entry["path"]
    )
    if (
        not target.exists()
        and local.exists()
        and local.stat().st_size == entry["size"]
        and digest(local) == entry["sha256"]
    ):
        os.link(local, target)
    else:
        run(
            [
                "gh",
                "release",
                "download",
                entry["release"],
                "--repo",
                "Shdakudow/ConvRML-Uncertainty",
                "--pattern",
                entry["name"],
                "--dir",
                target.parent,
                "--clobber",
            ],
            timeout=1800,
        )
        fetched = target.parent / entry["name"]
        if (
            fetched.stat().st_size != entry["size"]
            or digest(fetched) != entry["sha256"]
        ):
            raise ValueError("Checkpoint archive verification failed: " + entry["path"])
        fetched.replace(target)
    print("Verified archive checkpoint", entry["path"], flush=True)


def stage(epoch, required):
    missing = set(inventory(epoch, required)["missing"])
    if missing:
        selected = [entry for entry in required if entry["path"] in missing]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda entry: download(epoch, entry), selected))
        source = WORK / "milestone_cache" / f"epoch_{epoch:03d}"
        run(
            [
                "rsync",
                "-az",
                "-e",
                "ssh -o BatchMode=yes -i "
                + shlex.quote(str(Path.home() / ".ssh/id_ed25519")),
                str(source) + "/",
                HOST + ":" + REMOTE + f"/weights/epoch_{epoch:03d}/",
            ],
            timeout=7200,
        )
        if inventory(epoch, required)["missing"]:
            raise ValueError("Incomplete checkpoint transfer after verification")


def submit(epoch, tasks, kind="baseline"):
    response = remote("submit", dict(epoch=epoch, tasks=tasks, kind=kind))
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
