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
"""Data model and tag helper utilities for Optimization Memory entries."""

import re
from typing import Dict, List, Optional, Set, Tuple

_TRAILING_DIVIDER_RE = re.compile(r"\n+---\s*$")
_DESCRIPTION_BODY_RE = re.compile(r"\*\s*\*\*Description\*\*:\s*\n?(.*)",
                                  re.DOTALL)
_TAG_PAIR_RE = re.compile(r"\[([a-z]+):\s*([^\]]+)\]", re.IGNORECASE)


def extract_tag_tuples(
    tags_dict: Dict[str, List[str]], ) -> Set[Tuple[str, str]]:
    """Converts metadata dict to a set of uppercase (category, value) pairs.

  Time complexity: O(T) where T is the total number of tag values.

  Args:
    tags_dict: Dictionary mapping tag category name to list of values.

  Returns:
    Set of uppercase (category, value) tuples.
  """
    tag_set = set()
    for key, vals in tags_dict.items():
        for val in vals:
            tag_set.add((key.upper(), val.upper()))
    return tag_set


def parse_tags_input(tags_list: List[str]) -> Dict[str, List[str]]:
    """Parses a list of tag arguments into a structured metadata dictionary.

  Time complexity: O(T) where T is the number of raw tag items.

  Args:
    tags_list: List of raw tag strings (e.g., ['scope:OPTIMIZER', 'SHARDING']).

  Returns:
    Dictionary mapping lowercase category names to lists of tag values.
  """
    metadata: Dict[str, List[str]] = {}
    for item in tags_list:
        cleaned = item.strip()
        if not cleaned:
            continue
        if ":" in cleaned:
            cat, val = cleaned.split(":", 1)
            metadata.setdefault(cat.lower().strip(), []).append(val.strip())
        else:
            metadata.setdefault("topic", []).append(cleaned)
    return metadata


class Entry:
    """Represents a single entry in the Optimization Memory catalog.

  Attributes:
    id: Unique integer identifier (e.g., 1, 2, 3).
    score: Integer ranking score parsed from metadata (default: 1).
    metadata: Dictionary mapping tag category names to lists of tag values.
    description: Full markdown description body of the entry.
    source_file: Name of the origin file (e.g., 'optimization_memory.md').
  """

    def __init__(
        self,
        entry_id: int,
        metadata_dict: Dict[str, List[str]],
        description: str,
        source_file: str = "optimization_memory.md",
    ) -> None:
        """Initializes an Entry instance.

    Time complexity: O(1).

    Args:
      entry_id: Unique integer identifier for the entry.
      metadata_dict: Dictionary mapping tag categories to lists of values.
      description: Markdown description body of the entry.
      source_file: Origin filename.
    """
        self.id: int = int(entry_id)
        self.metadata: Dict[str, List[str]] = metadata_dict
        self.description: str = description
        self.source_file: str = source_file

        score = 1
        if "score" in self.metadata and self.metadata["score"]:
            try:
                score = int(self.metadata["score"][0])
            except ValueError:
                score = 1
        self.score: int = score

    def get_tag_values_set(self) -> Set[str]:
        """Returns all uppercase tag values and 'CATEGORY:VALUE' pairs.

    Time complexity: O(T) where T is the total number of tag values.

    Returns:
      Set of uppercase tag strings.
    """
        tag_set = set()
        for cat, vals in self.metadata.items():
            cat_upper = cat.upper()
            for val in vals:
                val_upper = val.upper()
                tag_set.add(val_upper)
                tag_set.add(f"{cat_upper}:{val_upper}")
        return tag_set

    def get_tag_tuples(self) -> Set[Tuple[str, str]]:
        """Returns uppercase (Category, Value) tuples for inverted index postings.

    Time complexity: O(T) where T is the total number of tag values.

    Returns:
      Set of (Category, Value) uppercase tuples.
    """
        return extract_tag_tuples(self.metadata)

    def has_any_tags(self, target_tags: List[str]) -> bool:
        """Checks whether the entry contains at least one tag from target_tags.

    Time complexity: O(T + Q) where T is tag values and Q is target tags count.

    Args:
      target_tags: List of query tag strings to match against.

    Returns:
      True if the intersection between entry tags and target_tags is non-empty.
    """
        if not target_tags:
            return True
        entry_tags = self.get_tag_values_set()
        return bool(entry_tags.intersection({t.upper() for t in target_tags}))

    def has_all_tags(self, target_tags: List[str]) -> bool:
        """Checks whether the entry contains all tags in target_tags.

    Time complexity: O(T + Q) where T is tag values and Q is target tags count.

    Args:
      target_tags: List of query tag strings to match against.

    Returns:
      True if target_tags is a subset of entry tags.
    """
        if not target_tags:
            return True
        entry_tags = self.get_tag_values_set()
        return {t.upper() for t in target_tags}.issubset(entry_tags)

    def to_markdown(self, override_id: Optional[int] = None) -> str:
        """Serializes the Entry into a standardized markdown block.

    Time complexity: O(T + L) where T is tag count and L is description length.

    Args:
      override_id: Optional integer ID to use instead of self.id.

    Returns:
      Standardized markdown entry string.
    """
        display_id = override_id if override_id is not None else self.id

        tags_formatted = [f"`[score: {self.score}]`"]
        standard_order = (
            "scope",
            "source",
            "status",
            "framework",
            "model",
            "hardware",
        )
        for cat in standard_order:
            if cat in self.metadata:
                for val in self.metadata[cat]:
                    tags_formatted.append(f"`[{cat}: {val}]`")

        for cat, vals in self.metadata.items():
            if cat not in standard_order and cat != "score":
                for val in vals:
                    tags_formatted.append(f"`[{cat}: {val}]`")

        meta_line = " ".join(tags_formatted)

        desc_clean = self.description.strip()
        match = _DESCRIPTION_BODY_RE.search(desc_clean)
        body_text = match.group(1).strip() if match else desc_clean
        body_text = _TRAILING_DIVIDER_RE.sub("", body_text).strip()

        indented_body = "\n".join(f"  {line}" if line.strip() else ""
                                  for line in body_text.splitlines())

        return (f"### Entry {display_id}\n* **Metadata**:"
                f" {meta_line}\n* **Description**:\n{indented_body}")
