"""Shared Slurm commands and durable job records for the submission scripts."""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import subprocess

METHODS = ("qutcc", "im2im")
SBATCH = "/usr/local/slurm/current/bin/sbatch"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Replace a complete record atomically; never expose partial JSON."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def submission_lock(run_root):
    """Serialize fresh and continuation submissions for the same run folder."""
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)
    with (run_root / "submit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def build_command(
    source, run_root, name, script, arguments, *, dependency=None, time_limit=None
):
    command = [
        SBATCH,
        "--parsable",
        f"--job-name={name}",
        f"--chdir={source.parents[1]}",
        f"--output={run_root}/logs/%j.out",
        f"--error={run_root}/logs/%j.err",
    ]
    if time_limit:
        command.append(f"--time={time_limit}")
    if dependency:
        if not str(dependency).isdigit():
            raise ValueError(f"Invalid dependency job ID: {dependency}")
        command.extend(
            [f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"]
        )
    return command + [str(source / "run.sbatch"), str(source / script)] + arguments


def save_records(run_root, record_path, records):
    write_json(record_path, records)
    canonical = run_root / "jobs.json"
    if record_path != canonical:
        current = read_json(canonical) if canonical.exists() else {}
        write_json(canonical, {**current, **records})


def submit_job(
    *,
    source,
    run_root,
    name,
    script,
    arguments,
    records,
    record_path,
    dependency=None,
    time_limit=None,
):
    """Reuse an identical recorded job or submit and immediately record one job.

    A rerun can finish a partially submitted plan without duplicating accepted
    jobs. Different commands for a recorded name require a new run directory.
    """
    command = build_command(
        source,
        run_root,
        name,
        script,
        arguments,
        dependency=dependency,
        time_limit=time_limit,
    )
    if name in records:
        record = records[name]
        if record["command"] != command or not str(record["job_id"]).isdigit():
            raise ValueError(
                f"Recorded submission differs for {name}; use a new run folder."
            )
        save_records(run_root, record_path, records)
        print(name, record["job_id"], "already recorded", flush=True)
        return record["job_id"]
    try:
        response = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Submission failed for {name}: {error.output.strip()}"
        ) from error
    job_id = response.strip().split(";")[0]
    if not job_id.isdigit():
        raise ValueError(
            f"Unexpected sbatch response for {name}: {response!r}; inspect Slurm before retrying."
        )
    records[name] = dict(job_id=job_id, command=command)
    if dependency:
        records[name]["dependency"] = dependency
    save_records(run_root, record_path, records)
    print(name, job_id, flush=True)
    return job_id


def require_same_training_code(repository, original):
    """Reject added, removed, or changed model/data/modification Python files."""
    paths = {Path("dataset.py")}
    for root in (repository, original):
        for directory in ("models", "convrml", "input_modifications"):
            paths.update(
                path.relative_to(root) for path in (root / directory).rglob("*.py")
            )
    for path in sorted(paths):
        new, old = repository / path, original / path
        if (
            not new.is_file()
            or not old.is_file()
            or new.read_bytes() != old.read_bytes()
        ):
            raise ValueError(f"Model/data/modification code changed: {path}")
