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
"""Unified CLI query and mutation tool for tpu-sharding Optimization Memory."""

import argparse
import os
import shutil
import sys
from typing import List, Optional

from entry import Entry, parse_tags_input
from index import EntryIndex
from persistence import load_index_from_markdown, save_index_to_markdown

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_REFERENCES_DIR = os.path.join(_MODULE_DIR, "..", "..", "references",
                               "tpu-sharding")
_PACKAGED_MEMORY_FILE_PATH = os.path.join(_REFERENCES_DIR,
                                          "optimization_memory.md")
_KNOB_FILE_PATH = os.path.join(_REFERENCES_DIR, "knob_wiki.md")

_MEMORY_FILE_ENV_VAR = "TPU_SHARDING_MEMORY_FILE"


def resolve_memory_file(explicit_path: Optional[str]) -> str:
    """Resolves the optimization memory file to operate on, seeding it if absent.

  Precedence is `--memory-file`, then the `TPU_SHARDING_MEMORY_FILE` environment
  variable, then the memory shipped inside the package. When the resolved path
  is a per-run copy that does not exist yet, it is seeded from the packaged
  memory so the run starts from the same catalog.

  Args:
    explicit_path: Value of `--memory-file`, or None when the flag is unset.

  Returns:
    Absolute path to the memory file reads and writes should target.
  """
    target = explicit_path or os.environ.get(_MEMORY_FILE_ENV_VAR)
    if not target:
        return _PACKAGED_MEMORY_FILE_PATH

    target = os.path.abspath(target)
    if os.path.exists(target):
        return target

    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(_PACKAGED_MEMORY_FILE_PATH):
        shutil.copyfile(_PACKAGED_MEMORY_FILE_PATH, target)
    return target


def load_catalog_index(target: str = "memory",
                       memory_file: Optional[str] = None) -> EntryIndex:
    """Loads an EntryIndex for the specified catalog target.

  Time complexity: O(L + N * T log N) where L is catalog file size, N is entry
  count, and T is tag count.

  Args:
    target: Target source catalog: 'memory', 'knob', or 'all'.
    memory_file: Optimization memory file to read; defaults to the packaged
      memory. The knob wiki is always read from the package, since it is
      read-only.

  Returns:
    Populated EntryIndex instance.
  """
    combined_index = EntryIndex()
    memory_path = memory_file or _PACKAGED_MEMORY_FILE_PATH

    if target in ("memory", "all"):
        if os.path.exists(memory_path):
            mem_index = load_index_from_markdown(memory_path)
            for entry in mem_index.entries.values():
                combined_index.add_entry(entry)

    if target in ("knob", "all"):
        if os.path.exists(_KNOB_FILE_PATH):
            knob_index = load_index_from_markdown(_KNOB_FILE_PATH)
            for entry in knob_index.entries.values():
                combined_index.add_entry(entry)

    return combined_index


def print_tag_categories(index: EntryIndex) -> None:
    """Prints all unique tag categories and their possible values.

  Time complexity: O(C * V log V) where C is categories and V is values count.

  Args:
    index: EntryIndex to read tags from.
  """
    tags_dict = index.get_all_tags_and_values()
    print("=" * 80)
    print("tpu-sharding Optimization Memory Tag Categories")
    print("=" * 80)
    for cat, values in sorted(tags_dict.items()):
        print(f"\n📂 [{cat.upper()}] ({len(values)} values):")
        for val in values:
            print(f"  • {val}")
    print("\n" + "=" * 80)


def print_tag_values(index: EntryIndex, category: str) -> None:
    """Prints all available values under a specific tag category.

  Time complexity: O(V) where V is the count of values in the category.

  Args:
    index: EntryIndex to read tags from.
    category: Name of the tag category.
  """
    tags_dict = index.get_all_tags_and_values()
    cat_key = category.strip().upper()

    if cat_key in tags_dict:
        values = tags_dict[cat_key]
        print(
            f"\nAvailable Values for Tag Category: '{cat_key}' ({len(values)}):"
        )
        for val in values:
            print(f"  • {val}")
    else:
        print(f"\n❌ Tag category '{category}' not found.")
        available = ", ".join(sorted(tags_dict.keys()))
        print(f"Available categories: {available}")


def print_entries(entries: List[Entry], query_type: str,
                  tags: List[str]) -> None:
    """Renders a list of matching entries in standardized markdown format.

  Time complexity: O(E * (T + D)) where E is entry count, T is tag count, and
  D is average description length.

  Args:
    entries: List of Entry objects to display.
    query_type: Description of query mode (e.g. 'ANY', 'ALL').
    tags: Query tag list.
  """
    print("=" * 80)
    print(f"Found {len(entries)} entries matching {query_type} of {tags}:")
    print("=" * 80 + "\n")

    if not entries:
        print("No matching entries found.")
        return

    for entry in entries:
        print(entry.to_markdown())
        print("\n" + "-" * 80 + "\n")


def execute_add_entry(tags: List[str], score: int, description: str,
                      memory_file: str) -> int:
    """Adds a new entry to the catalog and atomically persists to disk.

  Time complexity: O(L + N * T log N) for loading and saving index.

  Args:
    tags: List of raw tag strings (e.g. ['scope:OPTIMIZER', 'SHARDING']).
    score: Integer ranking score.
    description: Markdown description body.
    memory_file: Path to optimization memory file.

  Returns:
    The integer ID of the newly created entry.
  """
    index = load_index_from_markdown(memory_file)
    metadata = parse_tags_input(tags)
    metadata["score"] = [str(score)]

    new_entry = Entry(
        entry_id=0,
        metadata_dict=metadata,
        description=description,
        source_file=os.path.basename(memory_file),
    )

    new_id = index.add_entry(new_entry)
    save_index_to_markdown(index, memory_file)
    print(f"✅ Successfully created Entry {new_id} in {memory_file}.")
    return new_id


def execute_delete_entry(entry_id: int, memory_file: str) -> bool:
    """Deletes an entry by its integer ID and persists changes to disk.

  Time complexity: O(L + N * T log N) for loading and saving index.

  Args:
    entry_id: Integer ID of the entry to remove.
    memory_file: Path to optimization memory file.

  Returns:
    True if entry was deleted, False if not found.
  """
    index = load_index_from_markdown(memory_file)
    if not index.delete_entry(entry_id):
        print(f"❌ Entry {entry_id} not found in {memory_file}.")
        return False

    save_index_to_markdown(index, memory_file)
    print(f"✅ Successfully deleted Entry {entry_id} from {memory_file}.")
    return True


def execute_update_description(entry_id: int, new_description: str,
                               memory_file: str) -> bool:
    """Updates description of an entry and persists changes to disk.

  Time complexity: O(L + N * (T + D)) for loading and saving index.

  Args:
    entry_id: Integer ID of the entry to update.
    new_description: New description body.
    memory_file: Path to optimization memory file.

  Returns:
    True if updated, False if not found.
  """
    index = load_index_from_markdown(memory_file)
    if not index.update_entry_description(entry_id, new_description):
        print(f"❌ Entry {entry_id} not found in {memory_file}.")
        return False

    save_index_to_markdown(index, memory_file)
    print(f"✅ Successfully updated description for Entry {entry_id} in"
          f" {memory_file}.")
    return True


def execute_update_score(entry_id: int, new_score: int,
                         memory_file: str) -> bool:
    """Updates ranking score of an entry and persists changes to disk.

  Time complexity: O(L + N * T log N) for loading and saving index.

  Args:
    entry_id: Integer ID of the entry to update.
    new_score: New integer score value.
    memory_file: Path to optimization memory file.

  Returns:
    True if updated, False if not found.
  """
    index = load_index_from_markdown(memory_file)
    if not index.update_entry_score(entry_id, new_score):
        print(f"❌ Entry {entry_id} not found in {memory_file}.")
        return False

    save_index_to_markdown(index, memory_file)
    print(
        f"✅ Successfully updated score for Entry {entry_id} to {new_score} in"
        f" {memory_file}.")
    return True


def execute_update_tags(entry_id: int, tags: List[str],
                        memory_file: str) -> bool:
    """Updates metadata tags of an entry and persists changes to disk.

  Time complexity: O(L + N * T log N) for loading and saving index.

  Args:
    entry_id: Integer ID of the entry to update.
    tags: List of raw tag strings.
    memory_file: Path to optimization memory file.

  Returns:
    True if updated, False if not found.
  """
    index = load_index_from_markdown(memory_file)
    new_metadata = parse_tags_input(tags)

    if not index.update_entry_tags(entry_id, new_metadata):
        print(f"❌ Entry {entry_id} not found in {memory_file}.")
        return False

    save_index_to_markdown(index, memory_file)
    print(
        f"✅ Successfully updated tags for Entry {entry_id} in {memory_file}.")
    return True


def build_argument_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser for tpu-sharding Optimization Memory."""
    parser = argparse.ArgumentParser(
        description=
        ("Unified CLI Query & Management Helper for tpu-sharding Optimization"
         " Memory."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  1. List all tag categories:
     python3 query_tpu_sharding.py --list-tags

  2. List all values for a category:
     python3 query_tpu_sharding.py --tag-values topic

  3. Query entries matching ANY tag (OR query) with top-k:
     python3 query_tpu_sharding.py --has-any-tag LATENCY_HIDING SPARSECORE -k 5

  4. Query entries matching ALL tags (AND query):
     python3 query_tpu_sharding.py --has-all-tags OPTIMIZER MAXTEXT SHARDING

  5. Add a new entry:
     python3 query_tpu_sharding.py --add-entry \\
       --tags scope:OPTIMIZER framework:MAXTEXT \\
       topic:SHARDING model:QWEN_3.5 hardware:TPU_V6E \\
       --score 5 \\
       --description "Empirical takeaway description."

  6. Update description of an existing entry:
     python3 query_tpu_sharding.py --update-description 1 "Updated findings."
      """,
    )

    parser.add_argument(
        "--target",
        choices=["memory", "knob", "all"],
        default="memory",
        help="Target catalog source (default: memory).",
    )
    parser.add_argument(
        "--memory-file",
        default=None,
        help=("Optimization memory file to read and mutate, normally the "
              "per-run copy under the run's artifacts directory. Seeded from "
              "the packaged memory when it does not exist yet. Falls back to "
              f"${_MEMORY_FILE_ENV_VAR}, then to the packaged memory."),
    )

    # Query options
    query_group = parser.add_argument_group("Query Commands")
    query_group.add_argument(
        "--list-tags",
        action="store_true",
        help="List all tag categories and unique values.",
    )
    query_group.add_argument(
        "--tag-values",
        metavar="CATEGORY",
        help="List all values for a specific tag category.",
    )
    query_group.add_argument(
        "--has-any-tag",
        nargs="+",
        metavar="TAG",
        help="Query entries containing ANY of the specified tags (OR query).",
    )
    query_group.add_argument(
        "--has-all-tags",
        nargs="+",
        metavar="TAG",
        help="Query entries containing ALL of the specified tags (AND query).",
    )
    query_group.add_argument(
        "-k",
        type=int,
        default=10,
        help="Maximum number of entries to return (default: 10).",
    )

    # Mutation options
    mutation_group = parser.add_argument_group("Mutation Commands")
    mutation_group.add_argument(
        "--add-entry",
        action="store_true",
        help="Add a new entry to the optimization memory.",
    )
    mutation_group.add_argument(
        "--delete-entry",
        type=int,
        metavar="ENTRY_ID",
        help="Delete an entry by its integer ID.",
    )
    mutation_group.add_argument(
        "--update-description",
        nargs=2,
        metavar=("ENTRY_ID", "DESCRIPTION"),
        help="Update the description body of an entry.",
    )
    mutation_group.add_argument(
        "--update-score",
        nargs=2,
        metavar=("ENTRY_ID", "SCORE"),
        help="Update the ranking score of an entry.",
    )
    mutation_group.add_argument(
        "--update-tags",
        nargs="+",
        metavar="ARG",
        help=("Update the metadata tags of an entry (Usage: --update-tags"
              " <ENTRY_ID> <TAGS...>)."),
    )
    mutation_group.add_argument(
        "--tags",
        nargs="+",
        metavar="TAG",
        help="Tags to assign when adding an entry.",
    )
    mutation_group.add_argument(
        "--score",
        type=int,
        default=1,
        help="Score to assign when adding an entry (default: 1).",
    )
    mutation_group.add_argument(
        "--description",
        metavar="TEXT",
        help="Description text when adding an entry.",
    )

    return parser


def main() -> None:
    """CLI entry point for tpu-sharding catalog query and management."""
    parser = build_argument_parser()
    args = parser.parse_args()
    memory_file = resolve_memory_file(args.memory_file)

    # Handle mutations first
    if args.add_entry:
        if not args.tags or not args.description:
            print(
                "❌ Error: --add-entry requires both --tags and --description.")
            sys.exit(1)
        execute_add_entry(args.tags, args.score, args.description, memory_file)
        return

    if args.delete_entry is not None:
        execute_delete_entry(args.delete_entry, memory_file)
        return

    if args.update_description:
        try:
            entry_id = int(args.update_description[0])
        except ValueError:
            print("❌ Error: ENTRY_ID must be an integer.")
            sys.exit(1)
        execute_update_description(entry_id, args.update_description[1],
                                   memory_file)
        return

    if args.update_score:
        try:
            entry_id = int(args.update_score[0])
            score = int(args.update_score[1])
        except ValueError:
            print("❌ Error: ENTRY_ID and SCORE must be integers.")
            sys.exit(1)
        execute_update_score(entry_id, score, memory_file)
        return

    if args.update_tags:
        if len(args.update_tags) < 2:
            print(
                "❌ Error: --update-tags requires <ENTRY_ID> followed by tags.")
            sys.exit(1)
        try:
            entry_id = int(args.update_tags[0])
        except ValueError:
            print("❌ Error: ENTRY_ID must be an integer.")
            sys.exit(1)
        execute_update_tags(entry_id, args.update_tags[1:], memory_file)
        return

    # Handle queries
    index = load_catalog_index(args.target, memory_file)

    if args.list_tags:
        print_tag_categories(index)
    elif args.tag_values:
        print_tag_values(index, args.tag_values)
    elif args.has_any_tag:
        results = index.get_top_k_any(args.has_any_tag, k=args.k)
        print_entries(results, "ANY", args.has_any_tag)
    elif args.has_all_tags:
        results = index.get_top_k_all(args.has_all_tags, k=args.k)
        print_entries(results, "ALL", args.has_all_tags)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
