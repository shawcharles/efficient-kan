"""Report efficient-kan provenance for research manifests."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report efficient-kan provenance.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--source-checkout",
        action="store_true",
        help=(
            "Import the package from a source checkout's src/ tree. With the installed "
            "console command, this is resolved from --checkout-root or the current directory."
        ),
    )
    parser.add_argument(
        "--checkout-root",
        type=Path,
        default=None,
        help="Source checkout root to use for git and --source-checkout imports.",
    )
    return parser.parse_args(argv)


def _run_git(args: list[str], *, cwd: Path | None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_root(start: Path) -> Path | None:
    root = _run_git(["rev-parse", "--show-toplevel"], cwd=start)
    return Path(root).resolve() if root else None


def _source_paths(checkout_root: Path | None) -> tuple[Path | None, Path | None]:
    root = checkout_root.resolve() if checkout_root is not None else _git_root(Path.cwd())
    if root is None:
        return None, None
    return root, root / "src"


def _version_from_pyproject(checkout_root: Path | None) -> str | None:
    if checkout_root is None:
        return None

    pyproject = checkout_root / "pyproject.toml"
    if not pyproject.exists():
        return None

    match = re.search(
        r'(?m)^version\s*=\s*["\']([^"\']+)["\']',
        pyproject.read_text(encoding="utf-8"),
    )
    if match is None:
        return None
    return match.group(1)


def _distribution_version() -> str | None:
    try:
        return importlib.metadata.version("efficient-kan")
    except importlib.metadata.PackageNotFoundError:
        return None


def _import_package(*, source_checkout: bool, source_path: Path | None) -> tuple[Any, str]:
    if source_checkout:
        if source_path is None or not source_path.exists():
            raise SystemExit(
                "--source-checkout requires a checkout root containing a src/ directory."
            )
        sys.path.insert(0, str(source_path))
        return importlib.import_module("efficient_kan"), "source_checkout"

    try:
        return importlib.import_module("efficient_kan"), "environment"
    except ModuleNotFoundError:
        if source_path is None or not source_path.exists():
            raise
        sys.path.insert(0, str(source_path))
        return importlib.import_module("efficient_kan"), "source_fallback"


def collect_provenance(
    *,
    source_checkout: bool = False,
    checkout_root: Path | None = None,
) -> dict[str, Any]:
    root, source_path = _source_paths(checkout_root)
    efficient_kan, import_mode = _import_package(
        source_checkout=source_checkout,
        source_path=source_path,
    )
    status_short = _run_git(["status", "--short"], cwd=root)
    commit = _run_git(["rev-parse", "HEAD"], cwd=root)
    cuda_version = torch.version.cuda if torch.cuda.is_available() else None
    import_path = Path(efficient_kan.__file__).resolve()
    pyproject_version = _version_from_pyproject(root)
    distribution_version = _distribution_version()
    module_version = getattr(efficient_kan, "__version__", None)
    is_source_checkout_import = bool(source_path and source_path in import_path.parents)
    package_version = (
        (pyproject_version or module_version or distribution_version)
        if is_source_checkout_import
        else (distribution_version or module_version or pyproject_version)
    )

    return {
        "package": {
            "name": "efficient-kan",
            "version": package_version,
            "module_version": module_version,
            "distribution_version": distribution_version,
            "import_mode": import_mode,
            "import_path": str(import_path),
            "is_source_checkout_import": is_source_checkout_import,
        },
        "source_checkout": {
            "root": str(root) if root else None,
            "src_path": str(source_path) if source_path else None,
            "pyproject_version": pyproject_version,
        },
        "git": {
            "commit": commit,
            "dirty": bool(status_short),
            "status_short": status_short.splitlines() if status_short else [],
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": cuda_version,
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    provenance = collect_provenance(
        source_checkout=args.source_checkout,
        checkout_root=args.checkout_root,
    )

    if args.json:
        print(json.dumps(provenance, indent=2, sort_keys=True))
        return

    print("efficient-kan provenance")
    print(f"package_version: {provenance['package']['version']}")
    print(f"module_version: {provenance['package']['module_version']}")
    print(f"distribution_version: {provenance['package']['distribution_version']}")
    print(f"import_mode: {provenance['package']['import_mode']}")
    print(f"import_path: {provenance['package']['import_path']}")
    print(f"is_source_checkout_import: {provenance['package']['is_source_checkout_import']}")
    print(f"source_root: {provenance['source_checkout']['root']}")
    print(f"source_pyproject_version: {provenance['source_checkout']['pyproject_version']}")
    print(f"git_commit: {provenance['git']['commit']}")
    print(f"git_dirty: {provenance['git']['dirty']}")
    print(f"python: {provenance['python']['version']} ({provenance['python']['implementation']})")
    print(f"platform: {provenance['platform']['platform']}")
    print(f"pytorch: {provenance['pytorch']['version']}")
    print(f"cuda_available: {provenance['pytorch']['cuda_available']}")
    print(f"cuda_version: {provenance['pytorch']['cuda_version']}")


if __name__ == "__main__":
    main()
