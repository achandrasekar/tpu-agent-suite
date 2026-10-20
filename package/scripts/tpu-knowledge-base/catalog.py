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
"""The tables, and the walk from a key to a row.

Every `index/*.jsonl` is a table: one self-contained row per line, so a
changed entry is a one-line diff and `grep` is a working query. Rows carry
ids, not text. `template` names a file and is left that way; `prompt`
names a record in `prompts/prompt_table.jsonl`, and resolving it to the
instructions it holds is the one lookup this file does.

Nothing in this file fetches or writes anything.
"""

import functools
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = HERE.parent.parent
DATA_DIR = PACKAGE_DIR / "data" / "tpu-knowledge-base"
INDEX_DIR = DATA_DIR / "index"
PAGE_DIR = DATA_DIR / "pages"

# Rows that are not entries: a watchlist of repos, and a grouping table.
NOT_ENTRIES = {"pending.jsonl", "subjects.jsonl"}

# Where a row's evidence is. The field that is set is the kind.
SOURCES = ("url", "file", "page")

# `v6e-8` names a VM, `v6e` names the generation whose page answers for it.
ACCELERATOR_TYPE = re.compile(r"((?:tpu-?)?v?\d+[a-z]*)-\d+$")
# What a generation looks like, for the templated row: `v6e`, `tpu7x`.
GENERATION = re.compile(r"(?:tpu-?)?v?\d+[a-z]*")

# Tables whose pages the ranking reads as well as their rows. The wiki's
# are short, one subject each, and written to be found; the torch-tpu
# guides are long and range widely, and scored they took first place on
# questions they mention in passing. Measured in rank.py's PAGE_WEIGHT.
PAGE_TABLES = ("wiki.jsonl", )


@functools.cache
def _ids_in(names: tuple[str, ...]) -> frozenset[str]:
    found = set()
    for name in names:
        path = INDEX_DIR / name
        if path.is_file():
            found.update(
                json.loads(line)["id"]
                for line in path.read_text("utf-8").splitlines()
                if line.strip())
    return frozenset(found)


def page_scored(row: dict) -> bool:
    """Whether the ranking reads this row's page, and not just the row.

    Every `page` row, and every row in PAGE_TABLES.
    """
    return bool(row.get("page")) or row.get("id") in _ids_in(PAGE_TABLES)


def rows() -> list[dict]:
    """Every row of every table, in filename order."""
    found: list[dict] = []
    for path in sorted(INDEX_DIR.glob("*.jsonl")):
        if path.name in NOT_ENTRIES:
            continue
        for lineno, line in enumerate(path.read_text("utf-8").splitlines(), 1):
            if line.strip():
                try:
                    found.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{lineno}: {exc}") from exc
    return found


PROMPT_TABLE = DATA_DIR / "prompts" / "prompt_table.jsonl"


def prompts() -> dict[str, dict]:
    """The prompt table, keyed by the id a row's `prompt` names.

    Fails as `rows()` does, naming the line: a lookup resolves a kind
    before it serves a cached page, and a table that cannot be read must
    reach the caller as a ValueError, which the lookup degrades on, not
    a KeyError, which it does not.
    """
    if not PROMPT_TABLE.is_file():
        return {}
    found = {}
    for lineno, line in enumerate(
            PROMPT_TABLE.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            found[row["id"]] = row
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{PROMPT_TABLE.name}:{lineno}: {exc!r}") from exc
    return found


def instructions_template(prompt_id: str | None) -> str:
    """The instructions for reading this kind of source, `{keyword}` and
    all, or nothing.

    A row's `prompt` is an id in `prompts/prompt_table.jsonl`, not prose:
    `concept`, `tpu-spec`, `howto`. Handing the id to the model put the
    bare word "concept" at the end of the prompt, which is not an
    instruction -- it reads as a stray token, and every url row had one.

    What the record holds is worth the lookup: `tpu-spec` is where it says
    a figure must carry its denominator, because a vendor page quotes
    compute per chip and a roofline wants it per TensorCore.

    This is the text a cached page's identity is taken from. Every alias
    of a row shares one page, so what is fingerprinted must not vary with
    the alias the caller used; the keyword goes in only for the model.
    """
    found = prompts().get(prompt_id or "")
    if not found:
        return ""
    to_report = found.get("extract") or []
    parts = [
        found.get("lead", ""),
        ("Report:\n" + "\n".join(f"- {one}"
                                 for one in to_report)) if to_report else "",
        f"{found.get('trap_title', 'The trap')}: {found['trap']}"
        if found.get("trap") else "",
        f"How this fails: {found['fails_by']}"
        if found.get("fails_by") else "",
    ]
    # Sections on their own lines, bullets as bullets: joined with spaces the
    # four ran together into one paragraph and the last read as prose.
    return "\n\n".join(p for p in parts if p)


def with_keyword(template: str, keyword: str) -> str:
    """The template as the model reads it, for this keyword.

    `{keyword}` is substituted, not formatted -- the records are full of
    braces that are not fields.
    """
    return template.replace("{keyword}", keyword)


def instructions_for(prompt_id: str | None, keyword: str) -> str:
    """The instructions as the model reads them, for this keyword."""
    return with_keyword(instructions_template(prompt_id), keyword)


def normalize(key: str) -> str:
    """The form every spelling of one key shares.

    Case, spaces, underscores and leading dashes carry nothing, so they fold
    and nothing else does. This is spelling, not similarity: a key either
    normalizes onto one the table lists or it does not.
    """
    return re.sub(r"[\s_-]+", "-", key.strip().lower()).strip("-")


def spellings(written: str, generation: str) -> list[str]:
    """The ways one generation gets written, in the order to try them.

    `tpu-v5e`, `tpuv5e` and `v5e` are one chip, and which of them a row
    happens to list must not decide the answer. Trying only the bare form
    sent `v5e-4` past `hw-tpu-v5e`, which lists `tpu-v5e`, to the
    templated vendor url -- a page we ship, skipped for one we fetch.
    """
    found = [written, generation]
    if generation.startswith("v"):
        found.append(f"tpu-{generation}")
    return list(dict.fromkeys(found))


def find(table: list[dict], key: str) -> tuple[dict, str] | None:
    """The row a key names, and the keyword it is listed under.

    A caller asks for what it wants, not for the row that answers; a row
    lists every keyword that should reach it. Two things happen before a
    miss: an accelerator drops its chip count, and a generation no row
    lists falls to the row whose url is templated on it -- so a chip
    released tomorrow still resolves, and every key still names a row.
    """
    wanted = normalize(key)
    for row in table:
        if any(normalize(k) == wanted for k in row.get("keywords") or []):
            return row, wanted
    accelerator = ACCELERATOR_TYPE.match(wanted)
    written = accelerator.group(1) if accelerator else wanted
    generation = re.sub(r"^tpu-?(?=v\d)", "", written)
    if not GENERATION.fullmatch(generation):
        return None
    for spelling in spellings(written, generation):
        for row in table:
            if any(
                    normalize(k) == spelling
                    for k in row.get("keywords") or []):
                return row, spelling
    templated = next((r for r in table if "{keyword}" in r.get("url", "")),
                     None)
    return (templated, generation) if templated else None


def source_kind(row: dict) -> str:
    """Which of the three a row's evidence is. The field name is the tag."""
    for kind in SOURCES:
        if row.get(kind):
            return kind
    raise KeyError(f"{row.get('id')} has no source")


def contained(root: pathlib.Path, value: str, what: str) -> pathlib.Path:
    """`root/value`, or a refusal if it points outside `root`.

    One helper for both source kinds: when only `file` was checked, a
    `page` of `../../etc/passwd` resolved happily.
    """
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{what} escapes {root.name}: {value}")
    return resolved


def page_of(row: dict) -> pathlib.Path | None:
    """The shipped page a row points at, or None for a fetched one.

    `page` and `file` are both packaged content; they differ only in which
    root holds them.
    """
    if row.get("page"):
        return contained(PAGE_DIR, row["page"], f"{row.get('id')}: page")
    if row.get("file"):
        return contained(PACKAGE_DIR, row["file"], f"{row.get('id')}: file")
    return None


def front_field(text: str, name: str) -> str | None:
    """A value out of a page's header. Trailing comments are dropped."""
    for line in text.splitlines()[:20]:
        if line.startswith(f"{name}:"):
            return re.split(r"\s+#", line.split(":", 1)[1],
                            maxsplit=1)[0].strip()
    return None
