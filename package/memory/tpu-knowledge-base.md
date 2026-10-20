For a public fact about TPU hardware, runtimes or software -- a spec, a
default, a supported version, what a doc says -- do NOT answer from memory.
Route the question to the `tpu-knowledge-base` agent, general and multi-part
questions included, as the user asked it rather than broken into
sub-questions of your own. It splits the question itself, resolves each fact
against the source index installed at `{{data:tpu-knowledge-base/index}}`, and
returns the entry plus the public URL it came from. Send it a JSON object,
`{"query": "<the question>"}`; a plain-text request is rejected before the
agent starts. Add `entry_id` only with an id or keyword provenance returned
to you; a guessed one sends it to the nearest match, which may be a page it
has to build.
If it returns suggested keywords instead of sources, nothing was looked up:
those are not an answer.
Footnote every such claim with the URL it returns. If provenance reports that a
source does not state something, say so explicitly rather than filling the gap
yourself. Provenance reads only the sources its index names; it does not search
the web.
The playbooks and the Pallas kernel pages are pages at
`{{data:tpu-knowledge-base/pages}}/`; `kernel-optimization-index.md` there
routes the kernel pages. Cite the pages each claim relies on; if none covers a
topic, say so explicitly. The installed pages are read-only: never edit them.
