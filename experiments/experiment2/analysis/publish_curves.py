"""Collect five-epoch full-test curves and refresh the separate saved-image preview."""

import fcntl, json, pathlib, subprocess, time, shutil

WORK = pathlib.Path(__file__).resolve().parent
DEMO = pathlib.Path.home() / "ConvRML_Demo_Publish"
PACKAGE = (
    pathlib.Path.home()
    / "ConvRML_Code_Publish/experiments/experiment2_uncertainty_analysis_20261019"
)
REPORT = pathlib.Path.home() / "Downloads/ConvRML_Experiment2_Correlation_Analysis"
REMOTE = (
    "/home/jy2228/ConvRML/experiments/experiment2_uncertainty_analysis_20261019/results"
)
PYTHON = str(pathlib.Path.home() / "ConvRML/.venv/bin/python")
SSH = [
    "ssh",
    "-i",
    str(pathlib.Path.home() / ".ssh/id_ed25519"),
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=20",
    "jy2228@unicorn-login-02.coecis.cornell.edu",
]


def run(cmd, **kw):
    return subprocess.check_output(
        cmd, text=True, cwd=kw.pop("cwd", DEMO), **kw
    ).strip()


with (
    pathlib.Path.home() / ".local/share/convrml-raw-demo-20261019/publisher.lock"
).open("a") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(0)
    if run(["git", "diff", "--cached", "--name-only"]):
        raise SystemExit("Staged files exist; retry later")
    page = DEMO / "corrected-epoch50/analysis_metrics.json"
    d = json.loads(page.read_text())
    changed = False
    rml = (
        pathlib.Path.home()
        / ".local/share/convrml-paper50-demo-20261019/rml/report.json"
    )
    if rml.exists() and d["summary"]["cameras"]["rml"]["qutcc"]["status"] != "complete":
        run(
            [
                PYTHON,
                str(PACKAGE / "analyze_saved_examples.py"),
                "--output",
                str(REPORT),
            ],
            timeout=600,
        )
        run(
            [PYTHON, str(PACKAGE / "visualize.py"), "--results", str(REPORT)],
            timeout=600,
        )
        d["summary"] = json.loads((REPORT / "summary.json").read_text())
        rows = json.loads((REPORT / "rml_qutcc.json").read_text())["rows"]
        d["per_image"]["rml"]["qutcc"] = [
            {k: r[k] for k in ["id", "index", "lpips"]} | r["scales"]["0"] for r in rows
        ]
        for name in [
            "Experiment2_LPIPS_Uncertainty_Error.pdf",
            "per_image_metrics.csv",
        ]:
            shutil.copy2(REPORT / name, DEMO / "corrected-epoch50/analysis" / name)
        for name in [
            "rml_example_01.png",
            "rml_raw_vs_smoothed.png",
            "rml_smoothing.png",
        ]:
            shutil.copy2(
                REPORT / "figures" / name, DEMO / "corrected-epoch50/analysis" / name
            )
        changed = True
    code = (
        "import pathlib,json;p=pathlib.Path('"
        + REMOTE
        + "');out={};\nfor c in ['rml','diffuser']:\n out[c]={}\n for m in ['im2im','ensemble5','mc_005','qutcc']:\n  f=p/c/m/'summary.json'\n  if f.exists():out[c][m]=json.loads(f.read_text())\nprint(json.dumps(out))"
    )
    current = json.loads(run(SSH + ["python3 -"], input=code, timeout=60))
    full = d.setdefault("full_test", {})
    for c, models in current.items():
        for m, r in models.items():
            if m in full.get(c, {}):
                continue
            assert (
                r["n_images"] == 5000
                and r["epoch"] == 50
                and r["method"] == m
                and r["camera"] == c
            )
            row = {
                "status": "complete",
                "n_images": 5000,
                "epoch": 50,
                "lpips": r["metrics"]["lpips"],
                "scales": r["scales"],
                "test_id_sequence_sha256": r["test_id_sequence_sha256"],
            }
            peers = full.setdefault(c, {})
            assert all(
                v["test_id_sequence_sha256"] == row["test_id_sequence_sha256"]
                for v in peers.values()
            )
            peers[m] = row
            changed = True
            local = REPORT / "full_test" / c / m
            local.mkdir(parents=True, exist_ok=True)
            for filename in ["summary.json", "per_image.json"]:
                run(
                    [
                        "rsync",
                        "-az",
                        "-e",
                        "ssh -i "
                        + str(pathlib.Path.home() / ".ssh/id_ed25519")
                        + " -o BatchMode=yes",
                        f"jy2228@unicorn-login-02.coecis.cornell.edu:{REMOTE}/{c}/{m}/{filename}",
                        str(local) + "/",
                    ],
                    timeout=300,
                )
    curve_code = (
        "import pathlib,json;p=pathlib.Path('"
        + REMOTE
        + "');out={};\nfor c in ['rml','diffuser']:\n out[c]={}\n for m in ['im2im','ensemble5','mc_005','qutcc']:\n  rows=[]\n  for epoch in range(5,51,5):\n   f=p/c/m/('' if epoch==50 else ('epoch_%03d'%epoch))/'summary.json'\n   if f.exists():rows.append(json.loads(f.read_text()))\n  out[c][m]=rows\nprint(json.dumps(out))"
    )
    source_curves = json.loads(run(SSH + ["python3 -"], input=curve_code, timeout=60))
    curves = {}
    for camera, models in source_curves.items():
        curves[camera] = {}
        identity = None
        for method, reports in models.items():
            rows = []
            for report in reports:
                assert report["n_images"] == 5000 and report["epoch"] in range(5, 51, 5)
                if identity is None:
                    identity = report["test_id_sequence_sha256"]
                assert (
                    identity == report["test_id_sequence_sha256"]
                ), "Test IDs changed between milestones"
                rows.append(
                    dict(
                        epoch=report["epoch"],
                        n_images=5000,
                        lpips=report["metrics"]["lpips"],
                        pearson=report["scales"]["0"]["pearson"],
                        spearman=report["scales"]["0"]["spearman"],
                    )
                )
                dest = (
                    REPORT
                    / "full_test"
                    / camera
                    / method
                    / f"epoch_{report['epoch']:03d}"
                )
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            curves[camera][method] = rows
    if d.get("curves") != curves:
        d["curves"] = curves
        changed = True
    if not changed:
        print("No new metrics", time.ctime(), flush=True)
        raise SystemExit(0)
    page.write_text(json.dumps(d, separators=(",", ":")) + "\n")
    run(
        [
            "/Users/shdakudow/Downloads/ConvRML_MRI_Local/.venv/bin/python",
            "tools/verify_previous_demos.py",
        ],
        timeout=600,
    )
    run(
        [
            "git",
            "add",
            "corrected-epoch50/analysis_metrics.json",
            "corrected-epoch50/analysis",
        ]
    )
    run(
        [
            "git",
            "commit",
            "-m",
            "Update Experiment 2 LPIPS and uncertainty-error correlation results",
        ]
    )
    run(["git", "push", "origin", "main"], timeout=600)
    print("Published metric update", flush=True)
