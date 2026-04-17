#!/usr/bin/env python3
"""Interactively bump the DLUP version across the repository.

This tool updates a curated set of files that must stay in sync for releases:
build metadata, Python package version, docs, and example metadata.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PlannedEdit:
    path: Path
    description: str
    old: str
    new: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _validate_version(version: str) -> None:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.+-]+)?", version):
        raise ValueError(f"Invalid version '{version}'. Expected semver-like 'X.Y.Z' (optionally with -suffix/+meta).")


def _split_major_minor(version: str) -> str:
    match = re.match(r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)", version)
    if not match:
        raise ValueError(f"Invalid version '{version}'. Expected to start with X.Y.Z.")
    return f"{match.group('major')}.{match.group('minor')}"


def _read_current_dlup_version() -> str:
    init_py = REPO_ROOT / "aifo" / "dlup" / "dlup" / "__init__.py"
    match = re.search(r'^__version__\s*=\s*"(?P<version>[^"]+)"\s*$', _read_text(init_py), re.MULTILINE)
    if not match:
        raise ValueError(f"Could not determine current DLUP version from {init_py}")
    return match.group("version")


def _plan_regex_sub(
    *,
    path: Path,
    description: str,
    pattern: str,
    replacement: str,
    flags: int = 0,
    expected_matches: int | None = 1,
) -> PlannedEdit:
    content = _read_text(path)
    new_content, n = re.subn(pattern, replacement, content, flags=flags)
    if expected_matches is None:
        if n < 1:
            raise ValueError(f"{path}: expected at least 1 match for {description} (got {n})")
    elif n != expected_matches:
        raise ValueError(f"{path}: expected exactly {expected_matches} match(es) for {description} (got {n})")
    return PlannedEdit(path=path, description=description, old=content, new=new_content)


def _collect_plans(*, new_version: str) -> list[PlannedEdit]:
    short_version = _split_major_minor(new_version)
    plans: list[PlannedEdit] = []

    # Core build/version markers.
    plans.append(
        _plan_regex_sub(
            path=REPO_ROOT / "aifo" / "dlup" / "BUILD.bazel",
            description="Update DLUP_VERSION in aifo/dlup/BUILD.bazel",
            pattern=r'^(DLUP_VERSION\s*=\s*)"[^"]*"\s*$',
            replacement=f'\\g<1>"{new_version}"',
            flags=re.MULTILINE,
        )
    )
    plans.append(
        _plan_regex_sub(
            path=REPO_ROOT / "aifo" / "dlup" / "meson.build",
            description="Update project version in aifo/dlup/meson.build",
            pattern=r"^(\s*version\s*:\s*)'[^']*'\s*,\s*$",
            replacement=rf"\g<1>'{new_version}',",
            flags=re.MULTILINE,
        )
    )
    plans.append(
        _plan_regex_sub(
            path=REPO_ROOT / "aifo" / "dlup" / "pyproject.toml",
            description="Update version in aifo/dlup/pyproject.toml",
            pattern=r'^(version\s*=\s*)"[^"]*"\s*$',
            replacement=f'\\g<1>"{new_version}"',
            flags=re.MULTILINE,
        )
    )
    plans.append(
        _plan_regex_sub(
            path=REPO_ROOT / "aifo" / "dlup" / "include" / "dlup" / "constants.h",
            description="Update DLUP_VERSION in include/dlup/constants.h",
            pattern=r'^(#define\s+DLUP_VERSION\s+)"[^"]*"\s*$',
            replacement=f'\\g<1>"{new_version}"',
            flags=re.MULTILINE,
        )
    )
    plans.append(
        _plan_regex_sub(
            path=REPO_ROOT / "aifo" / "dlup" / "src" / "_libtiff_tiff_writer.cpp",
            description="Update DLUP_VERSION in src/_libtiff_tiff_writer.cpp",
            pattern=r'^(#define\s+DLUP_VERSION\s+)"[^"]*"\s*$',
            replacement=f'\\g<1>"{new_version}"',
            flags=re.MULTILINE,
        )
    )
    plans.append(
        _plan_regex_sub(
            path=REPO_ROOT / "aifo" / "dlup" / "dlup" / "__init__.py",
            description="Update __version__ in dlup/__init__.py",
            pattern=r'^(__version__\s*=\s*)"[^"]*"\s*$',
            replacement=f'\\g<1>"{new_version}"',
            flags=re.MULTILINE,
        )
    )

    # Docs.
    plans.append(
        _plan_regex_sub(
            path=REPO_ROOT / "aifo" / "dlup" / "docs" / "source" / "conf.py",
            description="Update release in docs/source/conf.py",
            pattern=r'^(release\s*=\s*)"[^"]*"\s*$',
            replacement=f'\\g<1>"{new_version}"',
            flags=re.MULTILINE,
        )
    )
    plans.append(
        _plan_regex_sub(
            path=REPO_ROOT / "aifo" / "dlup" / "docs" / "source" / "conf.py",
            description="Update version in docs/source/conf.py",
            pattern=r'^(version\s*=\s*)"[^"]*"\s*$',
            replacement=f'\\g<1>"{short_version}"',
            flags=re.MULTILINE,
        )
    )
    plans.append(
        _plan_regex_sub(
            path=REPO_ROOT / "aifo" / "dlup" / "docs" / "Doxyfile",
            description="Update PROJECT_NUMBER in docs/Doxyfile",
            pattern=r'^(PROJECT_NUMBER\s*=\s*)"?[0-9A-Za-z.+-]+"?\s*$',
            replacement=f'\\g<1>"{new_version}"',
            flags=re.MULTILINE,
        )
    )
    plans.append(
        _plan_regex_sub(
            path=REPO_ROOT / "aifo" / "dlup" / "docs" / "source" / "index.rst",
            description="Update citation version in docs/source/index.rst",
            pattern=r"(version\s*=\s*\{)[^}]+(\})",
            replacement=rf"\g<1>{new_version}\g<2>",
            flags=re.MULTILINE,
        )
    )

    # README and examples.
    plans.append(
        _plan_regex_sub(
            path=REPO_ROOT / "aifo" / "dlup" / "README.md",
            description="Update BibTeX version in README.md",
            pattern=r"(version\s*=\s*\{)[^}]+(\})",
            replacement=rf"\g<1>{new_version}\g<2>",
            flags=re.MULTILINE,
        )
    )
    plans.append(
        _plan_regex_sub(
            path=REPO_ROOT / "aifo" / "dlup" / "README.md",
            description="Update plain citation version in README.md",
            pattern=r"(Version\s+)[0-9A-Za-z.+-]+(\))",
            replacement=rf"\g<1>{new_version}\g<2>",
            flags=re.MULTILINE,
        )
    )
    plans.append(
        _plan_regex_sub(
            path=REPO_ROOT / "aifo" / "dlup" / "examples" / "python" / "files" / "dlup_annotation_test.xml",
            description="Update example Software tag in dlup_annotation_test.xml",
            pattern=r"(<Software>dlup\s+)[0-9A-Za-z.+-]+(</Software>)",
            replacement=rf"\g<1>{new_version}\g<2>",
            flags=re.MULTILINE,
        )
    )

    # Drop no-op plans: the regex already matched the expected number of times,
    # so the pattern is still valid; the substitution just happens to produce
    # identical content (e.g. patch bumps don't change the X.Y short version).
    return [p for p in plans if p.old != p.new]


def _print_plan(plans: list[PlannedEdit]) -> None:
    print("\nPlanned version updates:\n")
    for p in plans:
        print(f"- {p.path.relative_to(REPO_ROOT)}: {p.description}")
    print()


def _prompt(prompt: str) -> str:
    return input(prompt).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-version",
        help="New version to set. If omitted, you'll be prompted.",
    )
    args = parser.parse_args()

    current_version = _read_current_dlup_version()
    print(f"Current DLUP version: {current_version}")

    new_version = args.new_version or _prompt("Enter new version (e.g. 0.7.1): ")
    if not new_version:
        raise SystemExit("No version provided.")
    _validate_version(new_version)
    if new_version == current_version:
        raise SystemExit("New version matches current version; nothing to do.")

    plans = _collect_plans(new_version=new_version)
    _print_plan(plans)

    confirm = _prompt(f"Apply these {len(plans)} updates? [y/N]: ").lower()
    if confirm not in ("y", "yes"):
        print("Aborted; no files were changed.")
        return

    for p in plans:
        _write_text(p.path, p.new)

    print("\nDone. Updated files:")
    for p in plans:
        print(f"- {p.path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
