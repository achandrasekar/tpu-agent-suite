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
"""Markdown parser and atomic persistence layer for Optimization Memory."""

import os
import re
import tempfile
import textwrap
from typing import List

from entry import Entry
from index import EntryIndex

_ENTRY_HEADER_SPLIT_RE = re.compile(r"(?=(?:^|\n)###\s+)", re.IGNORECASE)
_HEADER_MATCH_RE = re.compile(
    r"^###\s+(?:💡\s*|⚙️\s*)?(?:Entry\s+(\d+)|([^\n:]+))", re.IGNORECASE)
_METADATA_LINE_RE = re.compile(r"\*\s*\*\*Metadata\*\*:\s*(.*)", re.IGNORECASE)
_TAG_PAIRS_RE = re.compile(r"\[([a-z0-9_-]+):\s*([^\]]+)\]", re.IGNORECASE)
_DESCRIPTION_SPLIT_RE = re.compile(r"\*\s*\*\*Description\*\*:\s*\n?(.*)",
                                   re.IGNORECASE | re.DOTALL)
_TRAILING_DIVIDER_RE = re.compile(r"\n+---\s*$", re.MULTILINE)


def parse_markdown_content(
        content: str,
        source_file: str = "optimization_memory.md") -> List[Entry]:
    """Parses raw markdown text into a list of Entry objects.

  Time complexity: O(L) where L is the total string length of content.

  Args:
    content: Raw markdown text containing entry blocks.
    source_file: Origin filename for provenance tracking.

  Returns:
    List of parsed Entry objects.
  """
    raw_chunks = _ENTRY_HEADER_SPLIT_RE.split(content)
    entries: List[Entry] = []

    auto_id_counter = 1
    for chunk in raw_chunks:
        trimmed = chunk.strip()
        if not trimmed or not trimmed.startswith("###"):
            continue

        lines = trimmed.splitlines()
        header_line = lines[0].strip()

        header_match = _HEADER_MATCH_RE.search(header_line)
        if header_match:
            if header_match.group(1):
                try:
                    entry_id = int(header_match.group(1))
                except ValueError:
                    entry_id = auto_id_counter
            else:
                entry_id = auto_id_counter
        else:
            entry_id = auto_id_counter

        auto_id_counter = max(auto_id_counter, entry_id + 1)

        metadata: dict[str, List[str]] = {}
        body_lines: List[str] = []

        for line in lines[1:]:
            meta_match = _METADATA_LINE_RE.match(line.strip())
            if meta_match:
                tag_str = meta_match.group(1)
                for cat, val in _TAG_PAIRS_RE.findall(tag_str):
                    metadata.setdefault(cat.lower().strip(),
                                        []).append(val.strip())
            else:
                body_lines.append(line)

        raw_body = "\n".join(body_lines).strip()
        desc_match = _DESCRIPTION_SPLIT_RE.search(raw_body)
        if desc_match:
            description = textwrap.dedent(desc_match.group(1)).strip()
        else:
            description = raw_body

        description = _TRAILING_DIVIDER_RE.sub("", description).strip()

        entry = Entry(
            entry_id=entry_id,
            metadata_dict=metadata,
            description=description,
            source_file=source_file,
        )
        entries.append(entry)

    return entries


def parse_markdown_file(file_path: str) -> List[Entry]:
    """Reads a markdown file from disk and parses its entries.

  Time complexity: O(L) where L is the file size in bytes.

  Args:
    file_path: Absolute or relative path to the markdown file.

  Returns:
    List of parsed Entry objects.
  """
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    base_name = os.path.basename(file_path)
    return parse_markdown_content(content, source_file=base_name)


def load_index_from_markdown(file_path: str) -> EntryIndex:
    """Loads an EntryIndex populated from a markdown catalog file.

  Time complexity: O(L + N * T log N) where L is file size, N is entry count,
  and T is tag count.

  Args:
    file_path: Path to the markdown file to load.

  Returns:
    EntryIndex instance containing all parsed entries.
  """
    index = EntryIndex()
    entries = parse_markdown_file(file_path)
    for entry in entries:
        index.add_entry(entry)
    return index


def extract_file_header(file_path: str) -> str:
    """Extracts the introductory header markdown above the first entry block.

  Time complexity: O(H) where H is the character length of the file header.

  Args:
    file_path: Path to the markdown file.

  Returns:
    Header markdown string up to the entry catalog divider.
  """
    if not os.path.exists(file_path):
        return "# tpu-sharding Optimization Memory\n\n---\n\n"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = _ENTRY_HEADER_SPLIT_RE.split(content)
    if chunks and not chunks[0].strip().startswith("###"):
        header = chunks[0].rstrip()
        if not header.endswith("---"):
            header = header + "\n\n---"
        return header + "\n\n"

    return "# tpu-sharding Optimization Memory\n\n---\n\n"


def save_index_to_markdown(index: EntryIndex, file_path: str) -> None:
    """Atomically writes an EntryIndex to disk as formatted markdown.

  Time complexity: O(N * (T + D)) where N is entry count, T is tag count, and
  D is average description length.

  Args:
    index: EntryIndex containing entries to serialize.
    file_path: Destination file path.
  """
    header = extract_file_header(file_path)

    sorted_entries = sorted(index.entries.values(), key=lambda e: e.id)
    entry_blocks = [entry.to_markdown() for entry in sorted_entries]

    catalog_body = "\n\n---\n\n".join(entry_blocks)
    full_content = header + catalog_body + "\n"

    target_dir = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(target_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile("w",
                                     dir=target_dir,
                                     delete=False,
                                     encoding="utf-8") as temp_file:
        temp_file.write(full_content)
        temp_path = temp_file.name

    os.replace(temp_path, file_path)
