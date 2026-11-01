"""Check source contracts without importing DataEvolver or executing workloads."""
import ast
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path


def check(root):
    root = Path(root)
    src = root / "src"
    errors = []
    if (root / ".git").exists():
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
        public_files = [root / name for name in names if name]
    else:
        public_files = [p for p in root.rglob("*") if p.is_file()]
    private_data = re.compile(
        r"/(?:Users|home)/[^/\s\"'<>]+|/(?:data\d+|datassd\d*|mnt|workspace|root)/|"
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
        r"\b(?:sk-proj-|sk-ant-|ghp_|github_pat_)[A-Za-z0-9_-]{18,}"
    )
    for path in public_files:
        if not path.is_file():
            continue
        if path.suffix.lower() in {".safetensors", ".ckpt", ".gguf", ".pt", ".pth", ".onnx", ".pem", ".key"}:
            errors.append(f"{path.relative_to(root)}: model weights or private-key files do not belong in public source")
        try:
            contents = path.read_text()
        except UnicodeDecodeError:
            continue
        match = private_data.search(contents)
        if match:
            line = contents.count("\n", 0, match.start()) + 1
            errors.append(f"{path.relative_to(root)}:{line}: private path or credential in public source")
    trees = {}
    for path in [*(src / "dataevolver").rglob("*.py"), *(root / "tests").rglob("*.py")]:
        try:
            trees[path] = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(root)}:{exc.lineno}: {exc.msg}")

    def module_exists(name):
        path = src.joinpath(*name.split("."))
        return path.with_suffix(".py").is_file() or path.is_dir()

    for path, tree in trees.items():
        package = path.relative_to(src).parent.parts if path.is_relative_to(src) else ()
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                parts = package[:len(package) - node.level + 1] if node.level else ()
                names = [".".join((*parts, *((node.module or "").split(".") if node.module else ())))]
            for name in names:
                if name.startswith("dataevolver.") and not module_exists(name):
                    errors.append(f"{path.relative_to(root)}:{node.lineno}: missing internal module {name}")
    for path in (root / "configs").rglob("*.json"):
        try:
            value = json.loads(path.read_text())
        except ValueError as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")
            continue
        if path.parent.name == "scene" and value.get("stage") == "blender":
            argv = value.get("argv", [])
            if argv and Path(argv[0]).name == "blender":
                prefix = argv[:argv.index("--python")] if "--python" in argv else argv
                if "--python-use-system-env" not in prefix or "--python-exit-code" not in prefix:
                    errors.append(f"{path.relative_to(root)}: Blender must enable PYTHONPATH and Python failure exit codes")
    scripts = tomllib.loads((root / "pyproject.toml").read_text())["project"]["scripts"]
    for command, target in scripts.items():
        module, _, function = target.partition(":")
        path = src.joinpath(*module.split(".")).with_suffix(".py")
        tree = trees.get(path)
        if tree is None or not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function for n in tree.body):
            errors.append(f"{command}: entry point {target} has no source function")
    canonical = "dataevolver.harness.cli:main"
    if any(scripts.get(command) != canonical for command in ("dataevolver", "dataevolver-harness")):
        errors.append("public Harness commands must share dataevolver.harness.cli:main")
    if {"dataevolver-production", "dataevolver-multimodal"} & scripts.keys():
        errors.append("retired standalone production commands must not bypass the Harness")
    return errors


if __name__ == "__main__":
    issues = check(Path(__file__).resolve().parents[4])
    if issues:
        print("\n".join(issues), file=sys.stderr)
        raise SystemExit(1)
    print("Source, configuration, import and CLI contracts passed; no workloads executed.")
