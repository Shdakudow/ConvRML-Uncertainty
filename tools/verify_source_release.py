"""Verify preserved source bytes and formatting equivalence without importing ML."""

import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def executable_ast(source):
    """Ignore only docstring whitespace adjusted by the source formatter."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                value = node.body[0].value
                value.value = " ".join(value.value.split())
    return ast.dump(tree, include_attributes=False)


def verify():
    manifest = json.loads(
        (ROOT / "provenance/unicorn_source_20261020.json").read_text()
    )
    archive_bytes = (ROOT / manifest["archive"]).read_bytes()
    assert hashlib.sha256(archive_bytes).hexdigest() == manifest["archive_sha256"]
    # Archive member names record the original cluster/publication layout.
    # Resolve repository-only moves without rewriting the captured archive.
    moves = json.loads(
        (ROOT / "provenance/layout_migration_20261021.json").read_text()
    )["moves"]

    def current_path(name):
        for old, new in moves.items():
            if name == old or name.startswith(old + "/"):
                return ROOT / (new + name[len(old) :])
        return ROOT / name

    python_count = 0
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        assert set(archive.getnames()) == set(manifest["files"])
        for name, record in manifest["files"].items():
            original = archive.extractfile(name).read()
            assert hashlib.sha256(original).hexdigest() == record["sha256"], name
            assert len(original) == record["bytes"], name
            published = current_path(name)
            assert published.is_file(), f"Missing published source: {name}"
            if published.suffix == ".py":
                before = executable_ast(original)
                after = executable_ast(published.read_bytes())
                assert before == after, f"Behavioral source change: {name}"
                python_count += 1
            else:
                assert published.read_bytes() == original, name
    print(
        f"Verified {len(manifest['files'])} archived files; "
        f"{python_count} published Python ASTs match the captured source."
    )


if __name__ == "__main__":
    verify()
