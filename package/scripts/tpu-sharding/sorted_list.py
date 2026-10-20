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
"""Self-balancing randomized binary search tree (Treap) for sorted rankings."""

import random
from typing import Generic, Iterator, Optional, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TreapNode(Generic[K, V]):
    """Node in a randomized binary search tree (Treap).

  Attributes:
    key: Sortable key used for binary search tree ordering.
    val: Associated value stored at the node.
    priority: Random float used to maintain the max-heap property.
    left: Left child in the Treap.
    right: Right child in the Treap.
    size: Total number of nodes in the subtree rooted at this node.
  """

    def __init__(self, key: K, val: V) -> None:
        """Initializes a TreapNode.

    Time complexity: O(1).

    Args:
      key: Sortable key used for ordering.
      val: Value stored at this node.
    """
        self.key: K = key
        self.val: V = val
        self.priority: float = random.random()
        self.left: Optional["TreapNode[K, V]"] = None
        self.right: Optional["TreapNode[K, V]"] = None
        self.size: int = 1

    def update_size(self) -> None:
        """Recalculates the subtree size based on left and right children.

    Time complexity: O(1).
    """
        self.size = 1
        if self.left:
            self.size += self.left.size
        if self.right:
            self.size += self.right.size


class SortedList(Generic[K, V]):
    """Self-balancing sorted collection backed by a randomized Treap.

  Maintains elements in strictly sorted key order with O(log N) expected time
  for insertion, deletion, and rank-based index selection.
  """

    def __init__(self) -> None:
        """Initializes an empty SortedList.

    Time complexity: O(1).
    """
        self.root: Optional[TreapNode[K, V]] = None

    def _split_equal_or_less(
        self, node: Optional[TreapNode[K, V]], key: Tuple[K, V]
    ) -> Tuple[Optional[TreapNode[K, V]], Optional[TreapNode[K, V]]]:
        """Splits subtree into keys <= key (left) and keys > key (right).

    Time complexity: O(log N) expected.

    Args:
      node: Subtree root node.
      key: (key, val) pivot tuple.

    Returns:
      Tuple of (left_subtree, right_subtree).
    """
        if not node:
            return None, None
        node_tuple = (node.key, node.val)
        if node_tuple <= key:
            left, right = self._split_equal_or_less(node.right, key)
            node.right = left
            node.update_size()
            return node, right
        else:
            left, right = self._split_equal_or_less(node.left, key)
            node.left = right
            node.update_size()
            return left, node

    def _split_less(
        self, node: Optional[TreapNode[K, V]], key: Tuple[K, V]
    ) -> Tuple[Optional[TreapNode[K, V]], Optional[TreapNode[K, V]]]:
        """Splits subtree into keys < key (left) and keys >= key (right).

    Time complexity: O(log N) expected.

    Args:
      node: Subtree root node.
      key: (key, val) pivot tuple.

    Returns:
      Tuple of (left_subtree, right_subtree).
    """
        if not node:
            return None, None
        node_tuple = (node.key, node.val)
        if node_tuple < key:
            left, right = self._split_less(node.right, key)
            node.right = left
            node.update_size()
            return node, right
        else:
            left, right = self._split_less(node.left, key)
            node.left = right
            node.update_size()
            return left, node

    def _merge(
        self,
        left: Optional[TreapNode[K, V]],
        right: Optional[TreapNode[K, V]],
    ) -> Optional[TreapNode[K, V]]:
        """Merges two treaps maintaining max-heap priority property.

    Time complexity: O(log N) expected.

    Args:
      left: Root of left Treap subtree.
      right: Root of right Treap subtree.

    Returns:
      Root of the merged Treap.
    """
        if not left or not right:
            return left or right
        if left.priority > right.priority:
            left.right = self._merge(left.right, right)
            left.update_size()
            return left
        else:
            right.left = self._merge(left, right.left)
            right.update_size()
            return right

    def add(self, key: K, val: V) -> None:
        """Inserts a (key, val) pair into the sorted collection.

    Time complexity: O(log N) expected.

    Args:
      key: Sortable key.
      val: Value associated with the key.
    """
        self.discard(key, val)
        new_node = TreapNode(key, val)
        key_tuple = (key, val)
        left_part, right_part = self._split_equal_or_less(self.root, key_tuple)
        self.root = self._merge(self._merge(left_part, new_node), right_part)

    def discard(self, key: K, val: V) -> None:
        """Removes a (key, val) pair from the collection if present.

    Time complexity: O(log N) expected.

    Args:
      key: Key of element to remove.
      val: Value of element to remove.
    """
        key_tuple = (key, val)
        left_part, right_part = self._split_less(self.root, key_tuple)
        _, right_prime = self._split_equal_or_less(right_part, key_tuple)
        self.root = self._merge(left_part, right_prime)

    def remove(self, key: K, val: V) -> None:
        """Alias for discard.

    Time complexity: O(log N) expected.

    Args:
      key: Key of element to remove.
      val: Value of element to remove.
    """
        self.discard(key, val)

    def get_at(self, index: int) -> Tuple[K, V]:
        """Retrieves the (key, val) tuple at the given 0-based sorted rank index.

    Time complexity: O(log N) expected.

    Args:
      index: 0-based rank index.

    Returns:
      The (key, val) tuple at that rank index.

    Raises:
      IndexError: If index is out of valid bounds.
    """
        if index < 0 or not self.root or index >= self.root.size:
            raise IndexError("Index out of range.")
        return self._select(self.root, index)

    def _select(self, node: TreapNode[K, V], index: int) -> Tuple[K, V]:
        """Selects element at 0-based index in subtree rooted at node.

    Time complexity: O(log N) expected.

    Args:
      node: Subtree root node.
      index: 0-based rank index.

    Returns:
      The (key, val) tuple at that rank.
    """
        left_size = node.left.size if node.left else 0
        if index < left_size:
            return self._select(node.left, index)
        elif index == left_size:
            return (node.key, node.val)
        else:
            return self._select(node.right, index - left_size - 1)

    def _inorder_traverse(
            self, node: Optional[TreapNode[K, V]]) -> Iterator[Tuple[K, V]]:
        """Traverses the tree in-order yielding elements in sorted key order.

    Time complexity: O(N).

    Args:
      node: Subtree root node.

    Yields:
      (key, val) tuples in ascending sorted key order.
    """
        if node:
            yield from self._inorder_traverse(node.left)
            yield (node.key, node.val)
            yield from self._inorder_traverse(node.right)

    def __iter__(self) -> Iterator[Tuple[K, V]]:
        """Iterates through all (key, val) pairs in ascending sorted key order.

    Time complexity: O(N).

    Returns:
      Iterator over (key, val) tuples in ascending sorted key order.
    """
        return self._inorder_traverse(self.root)

    def clear(self) -> None:
        """Removes all elements from the list.

    Time complexity: O(1).
    """
        self.root = None

    def __len__(self) -> int:
        """Returns the total number of elements stored in the list.

    Time complexity: O(1).

    Returns:
      Total count of elements.
    """
        return self.root.size if self.root else 0
