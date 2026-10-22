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


def check_shared_refactor(original, published):
    """Accept only the two documented structural changes, not arbitrary edits."""
    target = published.resolve().relative_to(ROOT).as_posix()
    entrypoints = {
        "shared/training/paper.py",
        "shared/evaluation/paper.py",
        "shared/evaluation/continuation.py",
        "shared/tests/paper_protocol.py",
    }
    if target in entrypoints:
        after = ast.parse(published.read_bytes())
        bootstrap = ast.parse(
            "sys.path.insert(0, str(Path(__file__).absolute().parent))"
        ).body[0]
        matches = [
            node
            for node in after.body
            if ast.dump(node, include_attributes=False)
            == ast.dump(bootstrap, include_attributes=False)
        ]
        assert len(matches) == 1
        after.body.remove(matches[0])
        imports = [
            node
            for node in after.body
            if isinstance(node, ast.Import)
            and len(node.names) == 1
            and node.names[0].name == "sys"
        ]
        assert len(imports) == 1
        after.body.remove(imports[0])
        if target == "shared/tests/paper_protocol.py":
            imports = [
                node
                for node in after.body
                if isinstance(node, ast.ImportFrom) and node.module == "pathlib"
            ]
            assert len(imports) == 1 and [alias.name for alias in imports[0].names] == [
                "Path"
            ]
            after.body.remove(imports[0])
        tree = ast.parse(original)
        if target == "shared/training/paper.py":
            # With a symlink entry point, resolve() would select the shared folder.
            # absolute() retains the experiment path used for outputs/source guards.
            assignments = [
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "ROOT"
                    for target in node.targets
                )
            ]
            assert len(assignments) == 1
            expected = ast.parse("Path(__file__).resolve().parent", mode="eval").body
            assert ast.dump(assignments[0].value) == ast.dump(expected)
            assignments[0].value = ast.parse(
                "Path(__file__).absolute().parent", mode="eval"
            ).body
        return executable_ast(ast.unparse(tree)) == executable_ast(ast.unparse(after))
    if (
        published
        == ROOT / "experiments/experiment2/four_models/models/convnext_norm_film.py"
    ):
        before, after = ast.parse(original), ast.parse(published.read_bytes())
        shared = ast.parse((ROOT / "shared/models/film.py").read_bytes())
        names = {"NormFiLM", "FiLMBackbone"}
        for name in names:
            old = next(
                node
                for node in before.body
                if isinstance(node, ast.ClassDef) and node.name == name
            )
            new = next(
                node
                for node in shared.body
                if isinstance(node, ast.ClassDef) and node.name == name
            )
            assert ast.dump(old, include_attributes=False) == ast.dump(
                new, include_attributes=False
            )
        before.body = [
            node
            for node in before.body
            if not (isinstance(node, ast.ClassDef) and node.name in names)
        ]
        imports = [
            node
            for node in after.body
            if isinstance(node, ast.ImportFrom) and node.module == "models._shared_film"
        ]
        assert len(imports) == 1 and {alias.name for alias in imports[0].names} == names
        after.body.remove(imports[0])
        return executable_ast(ast.unparse(before)) == executable_ast(ast.unparse(after))
    return False


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
                assert before == after or check_shared_refactor(
                    original, published
                ), f"Behavioral source change: {name}"
                python_count += 1
            else:
                assert published.read_bytes() == original, name
    print(
        f"Verified {len(manifest['files'])} archived files; "
        f"{python_count} published Python sources match, including reviewed shared-source moves."
    )


if __name__ == "__main__":
    verify()
