#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finds the XProf skill inside the installed xprof wheel and prints it.

Upstream ships `SKILL.md` plus its `references/` inside the `xprof` wheel,
co-versioned with the CLI they document. That co-versioning is the whole point:
guidance and binary cannot drift apart, because they are the same artifact. So the skill is read where the wheel put it, never copied: a copy is
a second artifact that can fall behind the binary, and one the installer has to
own, upgrade and remove.

It prints the skill rather than its path. The wheel sits outside the
workspace, and every harness but one asks the user before an agent reads
there, with nothing an installer can write to allow it ahead of time.
tpu-diagnostics has to run this script anyway, so the skill arrives through the
one call it already makes, the same way on every harness.

Three things it checks, because each fails quietly:

  * no XProf distribution at all;
  * `xprof` and `xprof-nightly` share a package directory and raise no pip
    conflict, so the skill found may describe a binary supplied by the other;
  * a release below this package's floor ships a skill for a CLI missing tools
    the package relies on, or no skill at all, so it is not enough.

A wheel at or above the floor without the skill is a broken install, reported
the same way.

Each is reported as an error ending in the command to ask the user to run. It
installs nothing, and neither may the agents that read it: which XProf an
environment holds is the user's call.

Usage:
    <xprof_interpreter> scripts/tpu-diagnostics/locate_xprof_skill.py
    <xprof_interpreter> scripts/tpu-diagnostics/locate_xprof_skill.py \
        --reference references/collect_profile.md

The first prints SKILL.md and the names of its references; the second prints
one reference, named as SKILL.md links it (`references/` and `.md` optional).
Exits 0 with the text, 1 when XProf is unusable, 2 for an unknown reference.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

# The skill directory inside the wheel.
SKILL_PACKAGE = "xprof.skills"
SKILL_DIRNAME = "xprof"

# The supported distribution: the official PyPI release. `xprof-nightly` was
# the only channel until LLO analysis, the roofline model and the skill shipped
# in a release (openxla/xprof#3840).
DISTRIBUTION = "xprof"
OTHER_DISTRIBUTION = "xprof-nightly"

# The package's floor. It must equal the `xprof>=` line in
# requirements-client.txt, which a test holds it to: this script runs under the
# XProf interpreter, from the installed package, where that file is not.
MINIMUM_VERSION = "2.24.0"
UPGRADE = (f"{sys.executable} -m pip install -U "
           f"'{DISTRIBUTION}>={MINIMUM_VERSION}'")


def installed_distributions() -> dict[str, str]:
    """Returns the installed xprof-providing distributions and their versions."""
    from importlib import metadata

    found = {}
    for name in (DISTRIBUTION, OTHER_DISTRIBUTION):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def _version_key(version: str) -> tuple[int, ...] | None:
    """A sort key for `X.Y.Z[aN][.postM]` versions, or None for any other.

    Releases and nightlies are re-cut as `.postM` (internal mirrors serve
    2.24.2a20260922.post1), and one must not read as uncomparable.
    """
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:a(\d+))?(?:\.post(\d+))?",
                         version)
    if not match:
        return None
    major, minor, patch, alpha, post = match.groups()
    # A pre-release sorts before the release it leads up to; a post-release
    # after the version it amends.
    return (int(major), int(minor), int(patch), 0 if alpha else 1,
            int(alpha or 0), int(post or 0))


def below_floor(version: str) -> bool | None:
    """Whether `version` is older than MINIMUM_VERSION; None if unknowable.

    `packaging` decides when the interpreter has it. A dedicated XProf venv may
    not, and skipping the check there would pass exactly the version it exists
    to catch, so the `X.Y.Z` scheme is compared directly instead.
    """
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        key, floor = _version_key(version), _version_key(MINIMUM_VERSION)
        return None if key is None or floor is None else key < floor
    try:
        return Version(version) < Version(MINIMUM_VERSION)
    except InvalidVersion:
        return None


def locate() -> dict[str, Any]:
    """Finds the skill inside the installed wheel. Pure data; no printing."""
    result: dict[str, Any] = {
        "skill_dir": None,
        "skill_md": None,
        "references": [],
        "version": None,
        "interpreter": sys.executable,
        "errors": [],
        "warnings": [],
    }

    dists = installed_distributions()
    result["version"] = dists.get(DISTRIBUTION)
    if not dists:
        result["errors"].append(
            "no xprof distribution is installed under "
            f"{sys.executable}. Ask the user to run: {UPGRADE}")
        return result

    if OTHER_DISTRIBUTION in dists:
        versions = ", ".join(f"{k}=={v}" for k, v in sorted(dists.items()))
        result["errors"].append(
            f"the `{OTHER_DISTRIBUTION}` distribution is installed "
            f"({versions}); `{DISTRIBUTION}` is the only supported channel. "
            "Both write into the same package directory with no pip conflict, "
            "so the skill found here may describe a binary supplied by the "
            f"other. Ask the user to run: {sys.executable} -m pip uninstall -y "
            f"{' '.join(sorted(dists))} && {UPGRADE}")
        return result

    version = dists[DISTRIBUTION]
    old = below_floor(version)
    if old:
        result["errors"].append(
            f"{DISTRIBUTION}=={version} is older than this package's floor "
            f"({MINIMUM_VERSION}), so its skill describes a CLI missing tools "
            f"the package relies on. Ask the user to run: {UPGRADE}")
        return result
    if old is None:
        result["warnings"].append(
            f"could not compare {DISTRIBUTION}=={version} with the floor "
            f"{MINIMUM_VERSION}; using it unchecked")

    from importlib.resources import files

    try:
        skill_dir = Path(str(files(SKILL_PACKAGE) / SKILL_DIRNAME))
    except (ImportError, TypeError):
        skill_dir = None
    skill_md = skill_dir / "SKILL.md" if skill_dir else None
    if skill_md is None or not skill_md.is_file():
        result["errors"].append(
            f"{DISTRIBUTION}=={version} has no {SKILL_PACKAGE}.{SKILL_DIRNAME}"
            f"/SKILL.md, so the install is incomplete. Ask the user to run: "
            f"{UPGRADE}")
        return result

    result["skill_dir"] = str(skill_dir)
    result["skill_md"] = str(skill_md)
    result["references"] = sorted(
        path.stem for path in (skill_dir / "references").glob("*.md"))
    return result


def reference_name(name: str) -> str:
    """`references/collect_profile.md`, as SKILL.md links it, to its stem."""
    name = name.strip()
    name = name.removeprefix("./").removeprefix("references/")
    return name.removesuffix(".md")


def render(report: dict[str, Any], reference: str | None) -> tuple[str, int]:
    """The text to print, and the exit status."""
    if report["errors"]:
        lines = ["XProf is unusable here, so there is no skill to read:"]
        lines += [f"- {error}" for error in report["errors"]]
        return "\n".join(lines), 1

    skill_dir = Path(report["skill_dir"])
    header = [
        f"XProf skill from {DISTRIBUTION}=={report['version']}, "
        f"the CLI under {report['interpreter']}.",
    ]
    header += [f"Warning: {warning}" for warning in report["warnings"]]

    if reference is None:
        script = Path(__file__).resolve()
        header += [
            "Print a reference SKILL.md links to with:",
            f"  {report['interpreter']} {script} --reference <name>",
            "References: " + ", ".join(report["references"]),
        ]
        body = report["skill_md"]
        title = "SKILL.md"
    else:
        name = reference_name(reference)
        if name not in report["references"]:
            return (f"No XProf reference named {reference!r}. References: " +
                    ", ".join(report["references"])), 2
        body = str(skill_dir / "references" / f"{name}.md")
        title = f"references/{name}.md"

    text = Path(body).read_text(encoding="utf-8")
    return "\n".join(header + ["", f"===== {title} =====", text]), 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reference",
        help="print this reference instead of SKILL.md, named as SKILL.md "
        "links it, e.g. references/collect_profile.md")
    args = parser.parse_args(argv)
    text, status = render(locate(), args.reference)
    print(text)
    return status


if __name__ == "__main__":
    sys.exit(main())
