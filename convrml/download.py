"""Download published checkpoints and verify their recorded size and SHA-256."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request

from .protocol import sha256

INDEX_PATH = Path(__file__).with_name("checkpoint_index.json")


def select_artifacts(dataset, method, epoch, member=None, members=5, *, index=None):
    registry = index or json.loads(INDEX_PATH.read_text())
    selected = [
        row
        for row in registry["artifacts"]
        if (row["dataset"], row["method"], row["epoch"]) == (dataset, method, epoch)
    ]
    if method == "ensemble":
        wanted = {member} if member is not None else set(range(members))
        selected = [row for row in selected if row["member"] in wanted]
        if {row["member"] for row in selected} != wanted or len(selected) != len(
            wanted
        ):
            raise ValueError(
                "The index does not contain exactly the requested ensemble members"
            )
    elif member is not None or len(selected) != 1:
        raise ValueError(
            "The index does not contain exactly the requested single-model checkpoint"
        )
    if any(not row["github_verified"] for row in selected):
        raise ValueError("A requested asset has not finished GitHub hash verification")
    return sorted(
        selected, key=lambda row: row["member"] if row["member"] is not None else -1
    )


def download_one(entry, output, *, repository="Shdakudow/ConvRML-Uncertainty"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / entry["name"]
    if target.exists():
        if target.stat().st_size == entry["size"] and sha256(target) == entry["sha256"]:
            return target
        raise FileExistsError(
            "Existing checkpoint has different bytes; choose another output folder"
        )
    with tempfile.TemporaryDirectory(prefix="download-", dir=output) as temporary:
        candidate = Path(temporary) / entry["name"]
        if shutil.which("gh"):
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    entry["release"],
                    "--repo",
                    repository,
                    "--pattern",
                    entry["name"],
                    "--dir",
                    temporary,
                ],
                check=True,
            )
        else:
            try:
                with (
                    urllib.request.urlopen(entry["url"], timeout=60) as response,
                    candidate.open("wb") as stream,
                ):
                    shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
            except urllib.error.HTTPError as error:
                raise RuntimeError(
                    f"Release access failed (HTTP {error.code}); use an authenticated GitHub CLI or obtain the indexed file manually"
                ) from None
        if (
            candidate.stat().st_size != entry["size"]
            or sha256(candidate) != entry["sha256"]
        ):
            raise ValueError(
                "Downloaded checkpoint does not match its recorded hash/size"
            )
        candidate.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["rml", "diffuser", "raw_clean", "raw_corrupted"],
        required=True,
    )
    parser.add_argument(
        "--method", choices=["im2im", "ensemble", "mc_dropout", "qutcc"], required=True
    )
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--member", type=int, choices=range(10))
    parser.add_argument("--members", type=int, choices=[5, 10], default=5)
    parser.add_argument("--output", type=Path, default=Path("weights"))
    parser.add_argument(
        "--list", action="store_true", help="Show the exact files without downloading"
    )
    args = parser.parse_args()
    selected = select_artifacts(
        args.dataset, args.method, args.epoch, args.member, args.members
    )
    for entry in selected:
        if args.list:
            print(
                json.dumps(
                    {key: entry[key] for key in ("name", "size", "sha256", "url")}
                )
            )
        else:
            print(download_one(entry, args.output), flush=True)


if __name__ == "__main__":
    main()
