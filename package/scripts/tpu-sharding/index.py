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
"""Inverted index engine for fast tag queries and entry management."""

import collections
import heapq
from typing import Dict, List, Set, Tuple

from entry import Entry, extract_tag_tuples
from sorted_list import SortedList


class PostingList:
    """Contains references to entries matching a specific tag query.

  Attributes:
    ids: Set of integer entry IDs matching this tag.
    by_score: Treap-backed SortedList keyed by (-score, entry_id) for descending
      score order.
  """

    def __init__(self) -> None:
        """Initializes an empty PostingList.

    Time complexity: O(1).
    """
        self.ids: Set[int] = set()
        self.by_score: SortedList[Tuple[int, int], int] = SortedList()


class EntryIndex:
    """Maintains an in-memory inverted index of Entry objects for fast queries.

  Attributes:
    entries: Dictionary mapping integer entry ID to Entry object.
    postings: Dictionary mapping normalized tag query strings to PostingList
      objects.
    tag_value_counts: Nested dictionary mapping tag category to value counts.
  """

    def __init__(self) -> None:
        """Initializes an empty EntryIndex.

    Time complexity: O(1).
    """
        self.entries: Dict[int, Entry] = {}
        self.postings: collections.defaultdict[str, PostingList] = (
            collections.defaultdict(PostingList))
        self.tag_value_counts: Dict[str, Dict[str, int]] = {}

    def _increment_tag(self, category: str, value: str) -> None:
        """Increments the frequency count for a (category, value) tag pair.

    Time complexity: O(1).

    Args:
      category: Tag category string.
      value: Tag value string.
    """
        if category not in self.tag_value_counts:
            self.tag_value_counts[category] = {}
        self.tag_value_counts[category][value] = (
            self.tag_value_counts[category].get(value, 0) + 1)

    def _decrement_tag(self, category: str, value: str) -> None:
        """Decrements the frequency count for a (category, value) tag pair.

    Time complexity: O(1).

    Args:
      category: Tag category string.
      value: Tag value string.
    """
        if (category in self.tag_value_counts
                and value in self.tag_value_counts[category]):
            self.tag_value_counts[category][value] -= 1
            if self.tag_value_counts[category][value] <= 0:
                del self.tag_value_counts[category][value]
                if not self.tag_value_counts[category]:
                    del self.tag_value_counts[category]

    def add_entry(self, entry: Entry) -> int:
        """Inserts a new Entry into the index and updates inverted postings.

    Time complexity: O(T log N) expected where T is tags count and N is posting
    list length.

    Args:
      entry: Entry instance to add.

    Returns:
      The integer ID assigned to the entry.
    """
        if entry.id is None or entry.id in self.entries or entry.id <= 0:
            next_id = max(self.entries.keys(), default=0) + 1
            entry.id = next_id

        entry_id = entry.id
        self.entries[entry_id] = entry

        score_key = (-entry.score, entry_id)
        for tag in entry.get_tag_values_set():
            self.postings[tag].ids.add(entry_id)
            self.postings[tag].by_score.add(score_key, entry_id)

        for cat, val in entry.get_tag_tuples():
            self._increment_tag(cat, val)

        return entry_id

    def delete_entry(self, entry_id: int) -> bool:
        """Removes an entry by its integer ID and updates all posting lists.

    Time complexity: O(T log N) expected where T is tags count and N is posting
    list length.

    Args:
      entry_id: Integer ID of the entry to remove.

    Returns:
      True if the entry was found and deleted, False otherwise.
    """
        if entry_id not in self.entries:
            return False

        entry = self.entries.pop(entry_id)
        score_key = (-entry.score, entry_id)

        for tag in entry.get_tag_values_set():
            if tag in self.postings:
                p_list = self.postings[tag]
                p_list.ids.discard(entry_id)
                p_list.by_score.discard(score_key, entry_id)
                if not p_list.ids:
                    del self.postings[tag]

        for cat, val in entry.get_tag_tuples():
            self._decrement_tag(cat, val)

        return True

    def update_entry_description(self, entry_id: int,
                                 new_description: str) -> bool:
        """Updates the markdown description for an existing entry.

    Time complexity: O(1).

    Args:
      entry_id: Integer ID of the entry.
      new_description: New description body.

    Returns:
      True if the entry was updated, False if not found.
    """
        if entry_id not in self.entries:
            return False
        self.entries[entry_id].description = new_description
        return True

    def update_entry_tags(self, entry_id: int,
                          new_metadata: Dict[str, List[str]]) -> bool:
        """Updates the metadata tags for an entry and rebuilds its postings.

    Time complexity: O(T log N) expected where T is tags count and N is posting
    list length.

    Args:
      entry_id: Integer ID of the entry.
      new_metadata: New metadata dictionary.

    Returns:
      True if the entry was updated, False if not found.
    """
        if entry_id not in self.entries:
            return False

        entry = self.entries[entry_id]
        old_score_key = (-entry.score, entry_id)

        old_tuples = entry.get_tag_tuples()
        new_tuples = extract_tag_tuples(new_metadata)

        # Clean up removed tags
        for cat, val in old_tuples - new_tuples:
            self._decrement_tag(cat, val)
            for tag_key in (val, f"{cat}:{val}"):
                if tag_key in self.postings:
                    p_list = self.postings[tag_key]
                    p_list.ids.discard(entry_id)
                    p_list.by_score.discard(old_score_key, entry_id)
                    if not p_list.ids:
                        del self.postings[tag_key]

        # Register newly added tags
        for cat, val in new_tuples - old_tuples:
            self._increment_tag(cat, val)
            for tag_key in (val, f"{cat}:{val}"):
                self.postings[tag_key].ids.add(entry_id)
                self.postings[tag_key].by_score.add(old_score_key, entry_id)

        if "score" not in new_metadata and entry.score is not None:
            new_metadata["score"] = [str(entry.score)]
        entry.metadata = new_metadata
        return True

    def update_entry_score(self, entry_id: int, new_score: int) -> bool:
        """Updates the ranking score of an entry and re-ranks it in all postings.

    Time complexity: O(T log N) expected where T is tags count and N is posting
    list length.

    Args:
      entry_id: Integer ID of the entry.
      new_score: New integer score value.

    Returns:
      True if the entry score was updated, False if not found.
    """
        if entry_id not in self.entries:
            return False

        entry = self.entries[entry_id]
        old_score = entry.score
        if old_score == new_score:
            return True

        old_key = (-old_score, entry_id)
        new_key = (-new_score, entry_id)

        for tag in entry.get_tag_values_set():
            if tag in self.postings:
                p_list = self.postings[tag]
                p_list.by_score.discard(old_key, entry_id)
                p_list.by_score.add(new_key, entry_id)

        entry.score = new_score
        entry.metadata["score"] = [str(new_score)]
        return True

    def get_top_k_any(self, query_tags: List[str], k: int) -> List[Entry]:
        """Retrieves top-k entries matching ANY tag in query_tags (OR query).

    Time complexity: O(k log Q) where Q is query tags count.

    Args:
      query_tags: List of query tag strings.
      k: Maximum number of ranked entries to return.

    Returns:
      List of matching Entry objects sorted in descending score order.
    """
        if not query_tags or k <= 0:
            return []

        posting_lists: List[PostingList] = []
        for tag_str in query_tags:
            tag_upper = tag_str.strip().upper()
            if tag_upper in self.postings:
                p_list = self.postings[tag_upper]
                if p_list.ids:
                    posting_lists.append(p_list)

        if not posting_lists:
            return []

        # Priority queue multi-way merge over posting lists
        # Heap stores: (neg_score, entry_id, list_idx, rank_pos)
        heap: List[Tuple[int, int, int, int]] = []
        for list_idx, p_list in enumerate(posting_lists):
            if len(p_list.by_score) > 0:
                score_key, eid = p_list.by_score.get_at(0)
                neg_score = score_key[0]
                heapq.heappush(heap, (neg_score, eid, list_idx, 0))

        results: List[Entry] = []
        seen_eids: Set[int] = set()

        while heap and len(results) < k:
            _, eid, list_idx, rank_pos = heapq.heappop(heap)
            if eid not in seen_eids:
                seen_eids.add(eid)
                results.append(self.entries[eid])

            next_pos = rank_pos + 1
            p_list = posting_lists[list_idx]
            if next_pos < len(p_list.by_score):
                next_item = p_list.by_score.get_at(next_pos)
                if next_item:
                    n_key, n_eid = next_item
                    heapq.heappush(heap, (n_key[0], n_eid, list_idx, next_pos))

        return results

    def get_top_k_all(self, query_tags: List[str], k: int) -> List[Entry]:
        """Retrieves top-k entries matching ALL tags in query_tags (AND query).

    Time complexity: O(min(|P|) * Q) where |P| is the rarest posting list length
    and Q is query tags count.

    Args:
      query_tags: List of query tag strings.
      k: Maximum number of ranked entries to return.

    Returns:
      List of matching Entry objects sorted in descending score order.
    """
        if not query_tags or k <= 0:
            return []

        normalized_tags = [t.strip().upper() for t in query_tags if t.strip()]
        if not normalized_tags:
            return []

        for tag in normalized_tags:
            if tag not in self.postings:
                return []

        # Pick the rarest tag (smallest posting list) as pivot
        rarest_tag = min(
            normalized_tags,
            key=lambda tag: len(self.postings[tag].ids),
        )
        other_tags = [t for t in normalized_tags if t != rarest_tag]

        results: List[Entry] = []
        p_list = self.postings[rarest_tag]

        for _, entry_id in p_list.by_score:
            entry = self.entries[entry_id]
            entry_tags = entry.get_tag_values_set()

            matches_all = True
            for tag in other_tags:
                if tag not in entry_tags:
                    matches_all = False
                    break

            if matches_all:
                results.append(entry)
                if len(results) == k:
                    break

        return results

    def get_all_tags_and_values(self) -> Dict[str, List[str]]:
        """Returns a dictionary of all unique tag categories and their values.

    Time complexity: O(C * V log V) where C is categories count and V is values
    per category.

    Returns:
      Dictionary mapping tag category names to sorted lists of unique values.
    """
        return {
            cat: sorted(list(vals.keys()))
            for cat, vals in self.tag_value_counts.items()
        }
