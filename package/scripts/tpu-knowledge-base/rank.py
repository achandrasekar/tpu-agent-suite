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
"""What the caller might have meant, when no keyword is what they typed.

`catalog.find` is exact: a key either normalizes onto one a row lists or
it does not. This is the other question, asked only after that one fails
-- "why is my flash attn slow" names no keyword and is still about
flash-attention-to-sdpa.

Okapi BM25 over each row's keywords, summary and note, plus its shipped
page as a second field. Lifted from index.py, which had it from
query_hw.py. Standard library only, as it was there.

A guess is never silent. `search` returns the score and the keyword it
matched on, `respellings` the words it swapped, and the caller says it was
a guess.
"""

import functools
import math
import pathlib
import re

_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ers", "er", "ed", "es",
             "e", "s")


@functools.cache
def stem(word: str) -> str:
    """`compile`, `compiled`, `compiler` and `compilation` as one word.

    Crude on purpose: it only has to map a word and its variants to the
    same string, and it does that on both sides of the comparison, so what
    the stem looks like does not matter.
    """
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", text.lower())


def tokens(text: str) -> list[str]:
    return [stem(w) for w in words(text)]


# Words a question is built from rather than about. "why is my matmul slow"
# asks about matmuls; left in, `why`, `is` and `my` score on whichever long
# page uses them most. Dropped from the query only, and only while
# something is left.
QUESTION_WORDS = frozenset("""
    an and any are as at be been but by can could did do does for from get
    got had has have here how in into is it its me my no not of on or our so
    some than that the then there these this those to was we were what when
    where which who why will with would you your
""".split())

# Prepositions and quantifiers, dropped with the question words. `after` in
# "instance norm after conv2d" matched the keyword `after_codegen.txt` and
# won the question for a notebook about XLA dumps. `all` stays: it is half
# of `all-to-all`, `all-gather` and `all-reduce`.
FILLER_WORDS = frozenset("""
    about after around before between during every many more much only out
    over per same
""".split())


def _synonyms(groups: list[str]) -> dict[str, tuple[str, ...]]:
    table: dict[str, tuple[str, ...]] = {}
    for group in groups:
        stems = [stem(w) for w in group.split()]
        for one in stems:
            table[one] = tuple(s for s in stems if s != one)
    return table


# Spellings of one thing that share no word: a query term brings the others
# in as extra terms. Only groups that moved the eval on their own. Tried
# alone, `hbm memory` and `bf16 bfloat16` cost first place on six and three
# queries, pulling in every page that says `memory` in passing.
SYNONYMS = _synonyms([
    "gemm matmul",
    "conv conv2d convolution",
    "remat rematerialization checkpointing",
    "hf huggingface",
])

# The shortest word respelt. At four, real words the tables happen not to
# use were "corrected" into ones they do -- `heap` to `head`, `mish` to
# `mesh` -- and cost clean queries their place.
#
# Respelling earns its keep on search_typos.jsonl: hit@5 0.876 against
# 0.795 without it, MRR 0.738 against 0.633. It costs nothing on the clean
# set, which has no typos to fix. It is still a guess -- `mistral` becomes
# `mixtral`, a different model -- so `respelt` names every word it changed.
TYPO_MIN_LENGTH = 5

_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def one_edit(word: str) -> set[str]:
    """Every string one letter dropped, added, changed or swapped.

    Generated and looked up, rather than compared against every word in the
    tables, so the cost grows with the word and not with the tables.
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    edits = {a + b[1:] for a, b in splits if b}
    edits.update(a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1)
    edits.update(a + c + b[1:] for a, b in splits if b for c in _ALPHABET)
    edits.update(a + c + b for a, b in splits for c in _ALPHABET)
    edits.discard(word)
    return edits


def page_text(path: pathlib.Path) -> str:
    try:
        return path.read_text("utf-8", errors="replace")
    except OSError:
        return ""


# The last table tokenized and what came of it. One search spends about
# 60ms reading and stemming the shipped pages, which is nothing beside the
# model call it exists to avoid -- but search_eval runs 566 searches over
# one table, where it is 25 seconds. Held by identity and only one deep, so
# a second table evicts the first rather than growing, and nothing here
# outlives the process or reaches disk.
_LAST: list = []


def page_row(row: dict) -> bool:
    """The rows whose page is scored when the caller does not say."""
    return bool(row.get("page"))


def _fields(table: list[dict], page_of,
            scored) -> tuple[list[tuple], dict[str, int]]:
    """Each row as the scorer sees it, and how often each word is written.

    A row is itself, its keywords, each keyword's stems, the stems of its
    keywords, summary and note together, and its page's stems -- for the
    rows `scored` admits. Which those are is the caller's to say: it is a
    fact about the tables, not about ranking.
    """
    if _LAST and _LAST[0] is table and _LAST[1] is scored:
        return _LAST[2], _LAST[3]
    docs = []
    used: dict[str, int] = {}
    for row in table:
        keywords = row.get("keywords") or []
        text = " ".join(
            [*keywords, row.get("summary", ""),
             row.get("note", "")])
        found = page_of(row) if scored(row) else None
        page = tokens(page_text(found)) if found and found.is_file() else []
        docs.append((row, keywords, [set(tokens(k))
                                     for k in keywords], tokens(text), page))
        for word in set(words(text)).union(
                words(page_text(found)) if found and found.is_file() else ()):
            used[word] = used.get(word, 0) + 1
    _LAST[:] = [table, scored, docs, used]
    return docs, used


def respelt(used: dict[str, int], query: str) -> tuple[str, list[tuple]]:
    """`query` with each word the tables never use swapped for its nearest.

    Words are matched whole, before stemming: `blocknig` is one swap from
    `blocking`, but its stem is nowhere near `block`. The replacement is
    the word the most rows use, so a typo lands on the common spelling
    rather than on a rare word that happens to be as close. Short words are
    left alone, and so is any word with a digit: one edit from `v8` is
    `v5`, and from `conv3d` is `conv1d`, both real and both wrong.
    """
    unknown = [
        w for w in words(query) if len(w) >= TYPO_MIN_LENGTH and w not in used
        and not re.search(r"[0-9]", w)
    ]
    if not unknown:
        return query, []
    fixed, swaps = query.lower(), []
    for word in dict.fromkeys(unknown):
        near = [w for w in one_edit(word) if w in used]
        if near:
            best = max(near, key=lambda w: (used[w], w))
            swaps.append((word, best))
            # Bounded by what `words` splits on, not by `\b`: `_` is a word
            # character to `\b`, so `flash_attnetion` would never be fixed.
            fixed = re.sub(rf"(?<![a-z0-9]){word}(?![a-z0-9])", best, fixed)
    return (fixed, swaps) if swaps else (query, [])


def respellings(table: list[dict],
                query: str,
                page_of,
                scored=page_row) -> list[tuple[str, str]]:
    """The words `search` swaps in `query`, as (typed, used) pairs."""
    if not table:
        return []
    return respelt(_fields(table, page_of, scored)[1], query)[1]


def query_terms(query: str) -> list[str]:
    """The query's stems, with question and filler words dropped, plus the
    synonyms they bring in -- each once, or a group would count double."""
    found = words(query)
    kept = [w
            for w in found if w not in QUESTION_WORDS | FILLER_WORDS] or found
    terms = [stem(w) for w in kept]
    extra = [s for t in terms for s in SYNONYMS.get(t, ()) if s not in terms]
    return terms + list(dict.fromkeys(extra))


# What a page is worth next to the keyword field, measured on
# tests/tpu-knowledge-base/search_eval.jsonl with the wiki's pages scored:
# hit@1 0.666 and MRR 0.754 at 0.5, against 0.641 and 0.736 at the 0.75
# this had when only `page` rows were scored. 0.5 is the best of 0.25 to
# 1.5; 0.625 is level and both ends are worse. On the 44 queries held out
# until it was chosen, hit@1 0.75 against 0.64. More pages in the field
# want less weight each: it was 1.5 when the field held 20.
PAGE_WEIGHT = 0.5

# The most one page can add. A handful of pages cover most of a kernel
# author's vocabulary, so uncapped the broadest took the top three for any
# question saying `kernel`, `memory` and `profile` -- hit@1 0.523 against
# 0.565 with the cap, on the same set.
PAGE_CAP = 10.0

K1, B = 1.5, 0.75


def _bm25(docs: list[list[str]], terms: list[str]) -> list[float]:
    """Okapi BM25 over one field."""
    present = [d for d in docs if d]
    if not present:
        return [0.0] * len(docs)
    avglen = sum(len(d) for d in present) / len(present)
    idf = {}
    for term in terms:
        n = sum(1 for d in present if term in d)
        idf[term] = math.log((len(present) - n + 0.5) / (n + 0.5) + 1.0)
    scores = []
    for doc in docs:
        score = 0.0
        for term in terms:
            found = doc.count(term)
            if found:
                score += idf[term] * (found * (K1 + 1)) / (
                    found + K1 * (1 - B + B * len(doc) / avglen))
        scores.append(score)
    return scores


def search(table: list[dict],
           query: str,
           page_of,
           limit: int = 5,
           scored=page_row) -> list[tuple[float, dict, str]]:
    """Rows the caller might have meant, best first, with the keyword hit.

    The page is scored as a field of its own, not pooled with the keywords:
    a page runs to thousands of words where a keyword list runs to twenty,
    so pooled it would set the average length and every keyword-only row
    would look short. Document frequency is counted over the pages alone
    for the same reason -- over every row, most of which have no page,
    `the` would look as rare as `retnet`.

    The keyword that comes back is a label, whichever of the row's shares
    the most terms with the query. It is not a promise the row answers to
    that spelling.
    """
    if not table:
        return []
    docs, used = _fields(table, page_of, scored)
    terms = query_terms(respelt(used, query)[0])
    if not terms:
        return []
    fields = _bm25([tokens for *_, tokens, _ in docs], terms)
    pages = _bm25([page for *_, page in docs], terms)

    scored = []
    for position, (row, keywords, split, _, _) in enumerate(docs):
        score = fields[position] + min(PAGE_WEIGHT * pages[position], PAGE_CAP)
        # A keyword is what the tables promise to answer to, so a hit there
        # outranks the same word buried in a note.
        for term in terms:
            if any(term in one for one in split):
                score += 1.5
        if score <= 0:
            continue
        best = max(zip(keywords, split),
                   key=lambda pair: sum(t in pair[1] for t in terms),
                   default=("", None))[0]
        # Position breaks ties, so the order is the same on every run: a
        # ranked list that reshuffles is not a ranking.
        scored.append((-score, position, row, best))
    scored.sort()
    return [(-score, row, keyword)
            for score, _, row, keyword in scored[:limit]]
