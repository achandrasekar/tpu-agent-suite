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
"""vllm_torchtpu patch: serve openai/gpt-oss-20b (MXFP4 checkpoint) on TPU.

A patch script rather than a fixed diff, so it survives plugin revisions:
every fix finds its own anchor in the installed source and decides, on its
own, one of three things.

- present: the change is already there, from upstream or from an earlier run
  of this script (it marks what it inserts, so the log says which). Skipped.
- missing, anchor found: applied.
- missing, anchor not found: the plugin changed in a way this script does not
  know. The whole patch fails naming the fix, and no file is written.

Every fix is decided before any file is written, so the plugin is either
patched in full or left as it was; a second run reports every fix present.

Files are located by module path under the directory holding
`vllm_torchtpu/`, never by importing them: importing the plugin starts the
TPU runtime, and the patch has to land before anything loads it. Both plugin
layouts are known: `layers/adapter` + `layers/core`, and the older
`layers/vllm` + `layers/common`.

    python3 gpt_oss_mxfp4.py [<plugin source root>]

`apply_vllm_patch.sh gpt_oss_mxfp4.py` runs it; that is how launch recipes and
the engine compatibility check call it. Why each fix exists, and how to verify
the server: tpu-knowledge-base keyword `vllm-gpt-oss-mxfp4`.

Exit codes, the applier's: 0 applied or already applied; 2 usage; 3 plugin
not found; 4 the patch does not apply to this plugin revision.
"""

import ast
import importlib.util
import os
import re
import sys
import tempfile

PATCH_NAME = "gpt_oss_mxfp4.py"
# Inserted with every change, so a later run can tell its own work from an
# upstream fix.
MARKER = "gpt-oss MXFP4 onboarding patch"
GPT_OSS_MXFP4 = "gpt_oss_mxfp4"

# Module candidates, newest layout first.
QUANT_METHODS = ("vllm_torchtpu.layers.core.quant_methods",
                 "vllm_torchtpu.layers.common.quant_methods")
QUANTIZATION = ("vllm_torchtpu.layers.adapter.quantization",
                "vllm_torchtpu.layers.vllm.quantization")
MXFP4 = ("vllm_torchtpu.layers.adapter.quantization.mxfp4",
         "vllm_torchtpu.layers.vllm.quantization.mxfp4")
PLATFORM = ("vllm_torchtpu.platforms.tpu_platform", )
ATTENTION = ("vllm_torchtpu.layers.adapter.attention",
             "vllm_torchtpu.layers.vllm.attention")

MOE_METHOD = "VllmMxfp4MoEMethod"
MXFP4_CONFIG = "VllmMxfp4Config"
GPT_OSS_CONFIG = "VllmGptOssMxfp4Config"


class Mismatch(Exception):
    """A fix's anchor is not in the installed plugin."""


class Present(Exception):
    """A fix's change is already in the installed plugin; says from where."""


def present(ours):
    """Present, from this patch when `ours`, else from upstream."""
    return Present("applied earlier by this patch" if ours else "upstream")


# --------------------------------------------------------------------------
# Locating and reading the plugin


def find_root():
    """The directory holding `vllm_torchtpu/`, found without importing it."""
    spec = importlib.util.find_spec("vllm_torchtpu")
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.dirname(
        os.path.abspath(list(spec.submodule_search_locations)[0]))


def module_file(root, module):
    """`module`'s source file under `root`, or None when it is not there."""
    base = os.path.join(root, *module.split("."))
    for path in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(path):
            return path
    return None


class Sources:
    """The plugin files the fixes edit, read once and written together."""

    def __init__(self, root):
        self.root = root
        self.original = {}
        self.text = {}

    def locate(self, candidates):
        """The first candidate module that exists, as a path; else Mismatch."""
        for module in candidates:
            path = module_file(self.root, module)
            if path:
                return path
        raise Mismatch("none of " + ", ".join(candidates) + " is in " +
                       self.root)

    def read(self, path):
        if path not in self.text:
            with open(path, encoding="utf-8") as f:
                self.original[path] = self.text[path] = f.read()
        return self.text[path]

    def changed(self):
        return [p for p in self.text if self.text[p] != self.original[p]]

    def write(self):
        """Writes every changed file, each replaced atomically."""
        for path in self.changed():
            mode = os.stat(path).st_mode
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path),
                                       prefix=".gpt_oss_mxfp4.")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(self.text[path])
                os.chmod(tmp, mode)
                os.replace(tmp, path)
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise


def rel(sources, path):
    return os.path.relpath(path, sources.root)


# --------------------------------------------------------------------------
# Text helpers


def one(pattern, text, what, flags=re.M):
    """The single match of `pattern` in `text`; Mismatch on none or many."""
    found = list(re.finditer(pattern, text, flags))
    if len(found) != 1:
        raise Mismatch(f"expected one {what}, found {len(found)}")
    return found[0]


def line_start(text, pos):
    return text.rfind("\n", 0, pos) + 1


def line_end(text, pos):
    end = text.find("\n", pos)
    return len(text) if end < 0 else end + 1


def offset(text, lineno, col=0):
    """Character offset of 1-based `lineno` and UTF-8 byte column `col`."""
    lines = text.splitlines(keepends=True)
    start = sum(len(line) for line in lines[:lineno - 1])
    prefix = lines[lineno - 1].encode("utf-8")[:col].decode("utf-8")
    return start + len(prefix)


def indent(text, pos):
    start = line_start(text, pos)
    return re.match(r"[ \t]*", text[start:]).group(0)


def append_block(text, block):
    """`text` with a module-level `block` appended at its end."""
    return text.rstrip("\n") + "\n\n\n" + block.strip("\n") + "\n"


def find_class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def find_method(tree, cls, name):
    node = find_class(tree, cls)
    if node is None:
        raise Mismatch(f"no class {cls}")
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    raise Mismatch(f"no {cls}.{name}")


def span(text, node):
    """(start, end) character offsets of `node`, decorators included."""
    decorators = getattr(node, "decorator_list", [])
    first = min([node.lineno] + [d.lineno for d in decorators])
    start = offset(text, first)
    return start, line_end(text, offset(text, node.end_lineno))


def defines(text, name):
    return re.search(rf"^{re.escape(name)}\s*=", text, re.M) is not None


# --------------------------------------------------------------------------
# The fixes. Each edit takes the current text of its file and returns the
# new text, or raises Present or Mismatch.


def edit_quant_name_constant(text):
    """`GPT_OSS_MXFP4` beside `MXFP4` in quant_methods."""
    if defines(text, "GPT_OSS_MXFP4"):
        raise present(
            re.search(rf"^GPT_OSS_MXFP4\s*=.*{MARKER}", text, re.M)
            is not None)
    anchor = one(r"^MXFP4\s*=\s*[\"']mxfp4[\"'][^\n]*\n", text,
                 'MXFP4 = "mxfp4" line')
    line = f'GPT_OSS_MXFP4 = "{GPT_OSS_MXFP4}"  # {MARKER}\n'
    return text[:anchor.end()] + line + text[anchor.end():]


def edit_platform_allowlist(text):
    """`gpt_oss_mxfp4` in `TpuPlatform.supported_quantization`."""
    tree = ast.parse(text)
    lists = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        if any(
                isinstance(t, ast.Name) and t.id == "supported_quantization"
                for t in targets) and isinstance(node.value, ast.List):
            lists.append(node.value)
    if len(lists) != 1:
        raise Mismatch("expected one supported_quantization list literal,"
                       f" found {len(lists)}")
    items = lists[0].elts
    values = [e.value for e in items if isinstance(e, ast.Constant)]
    if GPT_OSS_MXFP4 in values:
        start, end = span(text, lists[0])
        raise present(MARKER in text[start:end])
    if "mxfp4" not in values:
        raise Mismatch('supported_quantization does not list "mxfp4"')
    last = items[-1]
    pos = offset(text, last.end_lineno, last.end_col_offset)
    closing = offset(text, lists[0].end_lineno, lists[0].end_col_offset - 1)
    if not re.match(r"\s*,", text[pos:]):
        text = text[:pos] + "," + text[pos:]
        closing += 1 if closing >= pos else 0
    if last.end_lineno == lists[0].end_lineno:
        # A list on one line: extend it before its closing bracket, and mark
        # the line.
        text = text[:closing] + f' "{GPT_OSS_MXFP4}"' + text[closing:]
        eol = line_end(text, closing) - 1
        if "#" not in text[line_start(text, closing):eol]:
            text = text[:eol] + f"  # {MARKER}" + text[eol:]
        return text
    pad = indent(text, pos)
    pos = line_end(text, pos)
    return (text[:pos] + f"{pad}# {MARKER}: vLLM core renames a GPT-OSS\n"
            f'{pad}# checkpoint\'s "mxfp4" to "gpt_oss_mxfp4";'
            f" {GPT_OSS_CONFIG} serves it.\n"
            f'{pad}"{GPT_OSS_MXFP4}",\n' + text[pos:])


def edit_config_map(text, mxfp4_text):
    """`quant_methods.GPT_OSS_MXFP4: VllmGptOssMxfp4Config` in the config map."""
    if re.search(r"(quant_methods\.GPT_OSS_MXFP4|[\"']gpt_oss_mxfp4[\"'])\s*:",
                 text):
        raise present(f"{MARKER}: vLLM core's internal name" in text)
    if not re.search(rf"^class {GPT_OSS_CONFIG}\b", mxfp4_text, re.M):
        raise Mismatch(f"the mxfp4 module defines no {GPT_OSS_CONFIG} to map"
                       " the name to")
    entry = one(rf"^[ \t]*quant_methods\.MXFP4:\s*{MXFP4_CONFIG},[^\n]*\n",
                text, f"quant_methods.MXFP4: {MXFP4_CONFIG} map entry")
    pad = indent(text, entry.start())
    text = (text[:entry.end()] + f"{pad}# {MARKER}: vLLM core's internal name"
            " for GPT-OSS MXFP4 checkpoints.\n"
            f"{pad}quant_methods.GPT_OSS_MXFP4: {GPT_OSS_CONFIG},\n" +
            text[entry.end():])
    imp = one(
        r"^from (vllm_torchtpu\.layers\.\w+\.quantization\.mxfp4)"
        r" import\b", text, "import from the mxfp4 module")
    line = f"from {imp.group(1)} import {GPT_OSS_CONFIG}  # {MARKER}\n"
    return text[:imp.start()] + line + text[imp.start():]


def edit_config_class(text):
    """`VllmGptOssMxfp4Config`, whose get_name() is `gpt_oss_mxfp4`."""
    if (re.search(rf"^class {GPT_OSS_CONFIG}\b", text, re.M) or re.search(
            r"register_quantization_config\(\s*get_tpu_quant_method\("
            r"\s*(GPT_OSS_MXFP4|[\"']gpt_oss_mxfp4[\"'])", text)):
        raise present(f"{MARKER}: the config vLLM core" in text)
    if not re.search(rf"^class {MXFP4_CONFIG}\b", text, re.M):
        raise Mismatch(f"no class {MXFP4_CONFIG} to subclass")
    for name in ("register_quantization_config", "get_tpu_quant_method"):
        if not re.search(rf"\b{name}\b", text):
            raise Mismatch(f"the mxfp4 module does not import {name}")
    return append_block(
        text, f'''
# {MARKER}: the config vLLM core's "gpt_oss_mxfp4" name maps to.
@register_quantization_config(get_tpu_quant_method("{GPT_OSS_MXFP4}"))
class {GPT_OSS_CONFIG}({MXFP4_CONFIG}):
    """TPU config for vLLM core's "gpt_oss_mxfp4" (GPT-OSS MXFP4 checkpoints).

    Same TPU methods as {MXFP4_CONFIG}, but keeps get_name() ==
    "gpt_oss_mxfp4": vLLM's expert weight loader takes the GPT-OSS fused
    all-experts layout (gate_up/down blocks, block scales and biases loaded
    with shard_id=None) only under that name.
    """

    @classmethod
    def get_name(cls):
        return "{GPT_OSS_MXFP4}"
''')


def edit_forward_input_ids(text):
    """`input_ids` on the MoE forward, which vLLM passes since 0.22."""
    method = find_method(ast.parse(text), MOE_METHOD,
                         "_forward_monolithic_tpu")
    args = method.args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    if "input_ids" in names or args.kwarg is not None:
        header_end = offset(text, method.body[0].lineno)
        raise present(MARKER in text[offset(text, method.lineno):header_end])
    if (not args.args or args.args[-1].arg != "router_logits"
            or args.kwonlyargs or args.vararg or args.defaults):
        raise Mismatch(f"{MOE_METHOD}._forward_monolithic_tpu does not end"
                       " with router_logits")
    last = args.args[-1]
    pos = offset(text, last.end_lineno, last.end_col_offset)
    text = text[:pos] + ", input_ids: torch.Tensor | None = None" + text[pos:]
    eol = line_end(text, pos) - 1
    if "#" not in text[line_start(text, pos):eol]:
        text = text[:eol] + f"  # {MARKER}" + text[eol:]
    return text


def _process_weights(text):
    """(start, end) of VllmMxfp4MoEMethod.process_weights_after_loading."""
    return span(
        text,
        find_method(ast.parse(text), MOE_METHOD,
                    "process_weights_after_loading"))


_DEQUANT_W13 = r"^[ \t]*w13_weight = dequantize_mxfp4_packed\(\n"


def edit_expert_padding(text):
    """Zero-pad expert weights and biases to multiples of 128 for gmm_v2."""
    start, end = _process_weights(text)
    body = text[start:end]
    if "_gpt_oss_round_up(" in body:
        raise present(True)
    if re.search(
            r"(round_up|align_to|cdiv)\([^)\n]*128|//\s*128\)?\s*\*\s*128"
            r"|functional\.pad\(\s*w13|w13\w*\.new_zeros\(", body):
        raise present(False)
    if "def _gpt_oss_round_up(" in text:
        raise Mismatch("_gpt_oss_round_up is defined but not used")

    def sub(pattern, repl, what):
        nonlocal body
        found = one(pattern, body, what)
        body = body[:found.start()] + found.expand(repl) + body[found.end():]

    found = one(_DEQUANT_W13, body, "w13 dequantize_mxfp4_packed call")
    pad = indent(body, found.start())
    body = (body[:found.start()] + f"{pad}device = layer.w13_weight.device"
            f"  # {MARKER}\n" + body[found.start():])
    sub(r"\)\.to\(layer\.w13_weight\.device\)\n", ")\n",
        "w13 dequantized weight moved to the device")
    sub(r"\)\.to\(layer\.w2_weight\.device\)\n", ")\n",
        "w2 dequantized weight moved to the device")
    found = one(
        r"^([ \t]*)w2_weight = w2_weight\.transpose\(1, 2\)"
        r"\.contiguous\(\)[^\n]*\n", body, "w2 transpose")
    p = found.group(1)
    body = body[:found.end()] + f"""
{p}# {MARKER}: gmm_v2 rounds tile_k/tile_n up to a lane
{p}# multiple (128), masks only rhs in the last K tile and runs with bounds
{p}# checks disabled, so a K that is not a multiple of 128 (GPT-OSS: hidden =
{p}# intermediate = 2880) multiplies whatever lies past the end of lhs (NaN),
{p}# and a fused-activation N of 2 * 2880 fails its 2 * num_lanes check.
{p}# Zero-pad hidden (K of w13, N of w2) and each gate/up half (N of w13, K
{p}# of w2) to 128; fused_moe_func pads x to w13's K and slices the output
{p}# back to hidden.
{p}hidden = w13_weight.shape[1]
{p}half = w13_weight.shape[2] // 2
{p}hidden_p = _gpt_oss_round_up(hidden, 128)
{p}half_p = _gpt_oss_round_up(half, 128)
{p}num_experts = w13_weight.shape[0]
{p}w13_p = w13_weight.new_zeros((num_experts, hidden_p, 2 * half_p))
{p}w13_p[:, :hidden, :half] = w13_weight[..., :half]
{p}w13_p[:, :hidden, half_p:half_p + half] = w13_weight[..., half:]
{p}w2_p = w2_weight.new_zeros((num_experts, half_p, hidden_p))
{p}w2_p[:, :half, :hidden] = w2_weight
{p}w13_weight, w2_weight = w13_p.to(device), w2_p.to(device)
""" + body[found.end():]
    sub(r"^([ \t]*)w13_bias = layer\.w13_bias\.data\n",
        r'\1w13_bias = layer.w13_bias.data.to("cpu", torch.float32)\n',
        "w13 bias read")
    found = one(
        r"^([ \t]*)layer\.w13_bias = torch\.nn\.Parameter\(\n"
        r"[ \t]*w13_bias\.unsqueeze\(1\)\.to\(torch\.float32\),\n", body,
        "w13 bias parameter")
    p = found.group(1)
    body = body[:found.start()] + (
        f"{p}b13 = w13_bias.new_zeros((num_experts, 2 * half_p))  # {MARKER}\n"
        f"{p}b13[:, :half] = w13_bias[:, :half]\n"
        f"{p}b13[:, half_p:half_p + half] = w13_bias[:, half:]\n" +
        found.group(0).replace(
            "w13_bias.unsqueeze(1).to(torch.float32)",
            "b13.unsqueeze(1).to(device)")) + body[found.end():]
    found = one(
        r"^([ \t]*)layer\.w2_bias = torch\.nn\.Parameter\(\n"
        r"[ \t]*layer\.w2_bias\.data\.unsqueeze\(1\)\.to\(torch\.float32\),\n",
        body, "w2 bias parameter")
    p = found.group(1)
    body = body[:found.start()] + (
        f'{p}w2_bias = layer.w2_bias.data.to("cpu", torch.float32)'
        f"  # {MARKER}\n"
        f"{p}b2 = w2_bias.new_zeros((num_experts, hidden_p))\n"
        f"{p}b2[:, :hidden] = w2_bias\n" + found.group(0).replace(
            "layer.w2_bias.data.unsqueeze(1).to(torch.float32)",
            "b2.unsqueeze(1).to(device)")) + body[found.end():]
    text = text[:start] + body + text[end:]
    return append_block(
        text, f'''
def _gpt_oss_round_up(x: int, a: int) -> int:  # {MARKER}
    return (x + a - 1) // a * a
''')


def edit_unloaded_tail(text):
    """Zero the per-rank intermediate tail vLLM's gpt-oss loader never writes."""
    if "_zero_gpt_oss_unloaded_intermediate(" in text:
        raise present(True)
    start, end = _process_weights(text)
    body = text[start:end]
    found = one(_DEQUANT_W13, body, "w13 dequantize_mxfp4_packed call")
    if "torch" not in re.findall(r"^import (\w+)", text, re.M):
        raise Mismatch("the mxfp4 module does not import torch")
    pad = indent(body, found.start())
    body = (body[:found.start()] +
            f"{pad}_zero_gpt_oss_unloaded_intermediate(layer)  # {MARKER}\n" +
            body[found.start():])
    text = text[:start] + body + text[end:]
    return append_block(
        text, f'''
def _zero_gpt_oss_unloaded_intermediate(layer) -> None:  # {MARKER}
    """Zeroes the intermediate tail vLLM's gpt-oss loader never writes.

    Mxfp4MoEMethod.create_weights rounds each rank's intermediate slice up
    (2880 / 8 = 360 -> 384 under a flattened MoE TP of 8), and
    GptOssModel._load_weights_mxfp4 copies only the rank's real slice (the
    last rank gets 192 of its 384). vLLM relies on create_weights'
    torch.zeros for the rest, but the plugin's expert stager loads into
    reused torch.empty host buffers, so the tail can hold stale bytes: NaN
    e8m0 scales or NaN biases there turn the rank's partial sum, and after
    the MoE all-reduce every token, into NaN. The slice bounds below are the
    loader's. Ranks without a tail, and layers that are not GPT-OSS MXFP4,
    are left alone.
    """
    quant_config = getattr(layer, "quant_config", None)
    if quant_config is None or quant_config.get_name() != "{GPT_OSS_MXFP4}":
        return
    moe_config = getattr(layer, "moe_config", None)
    parallel = getattr(moe_config, "moe_parallel_config", None)
    full = getattr(moe_config, "intermediate_size", None)
    if parallel is None or not full:
        return
    block = 32  # OCP MX block size
    per_rank = -(-(full // block) // parallel.tp_size) * block
    start = parallel.tp_rank * per_rank
    loaded = max(0, min(start + per_rank, full) - start)
    # (parameter, axis of the intermediate dimension, first unloaded index)
    tails = (
        ("w13_weight", 1, 2 * loaded),
        ("w13_weight_scale", 1, 2 * loaded),
        ("w13_bias", 1, 2 * loaded),
        ("w2_weight", -1, loaded // 2),  # two fp4 values per byte
        ("w2_weight_scale", -1, loaded // block),
    )
    for name, axis, first in tails:
        param = getattr(layer, name, None)
        if param is None or first >= param.shape[axis]:
            continue
        # On TPU an in-place slice write rewrites the whole tensor once per
        # slice; zero a host copy and upload it once instead.
        device = param.data.device
        host = param.data.to("cpu")
        host.narrow(axis, first, host.shape[axis] - first).zero_()
        param.data = host.to(device)
''')


def edit_hd64_attention_sinks(text):
    """Keep head_dim 64 layers on the hd64 kernel, which has attention sinks."""
    if "own_hd64_pool" in text:
        raise present(MARKER in text)
    if "pool_head_dim" not in text:
        raise Present("not needed: this revision does not widen q/k/v to the"
                      " KV pool")
    found = one(r"^([ \t]*)if pool_head_dim > self\.head_size:[ \t]*\n", text,
                "`if pool_head_dim > self.head_size:` line")
    for name in ("TPU_HEAD_SIZE_ALIGNMENT", "cdiv", "_pool_is_seq_along_lane",
                 "num_kv_heads"):
        if not re.search(rf"\b{name}\b", text):
            raise Mismatch(f"the attention module has no {name}")
    p = found.group(1)
    return text[:found.start()] + f"""\
{p}# {MARKER}: a head_dim == 64 layer allocates its own
{p}# pool in the hd64 layout, whose last axis is already padded to
{p}# TPU_HEAD_SIZE_ALIGNMENT (K/V packed along head_dim, num_kv_heads rows).
{p}# That pool is not a wider page: widening q/k/v to 128 would steer
{p}# sharded_ragged_paged_attention off the hd64 kernel, the only one with
{p}# attention sinks (GPT-OSS). With one KV head per device, packing pads
{p}# both layouts to the same row count, so a layer with sinks, which only
{p}# the hd64 kernel can run, is taken to own its pool as well.
{p}own_hd64_pool = (
{p}    self.head_size == 64
{p}    and kv_cache.ndim != 4
{p}    and not self._pool_is_seq_along_lane
{p}    and pool_head_dim
{p}    == cdiv(self.head_size, TPU_HEAD_SIZE_ALIGNMENT) * TPU_HEAD_SIZE_ALIGNMENT
{p}    and (
{p}        kv_cache.shape[-3] * kv_cache.shape[-2] < 2 * self.num_kv_heads
{p}        or getattr(self, "sinks", None) is not None
{p}    )
{p})
{p}if pool_head_dim > self.head_size and not own_hd64_pool:
""" + text[found.end():]


# Each fix: (name, what it does, [(file candidates, edit)]). An edit whose
# text depends on another file reads it through `sources` at call time.
def fixes(sources):
    mxfp4 = lambda: sources.read(sources.locate(MXFP4))  # noqa: E731
    return [
        ("gpt-oss-config-class",
         "VllmGptOssMxfp4Config, whose get_name() matches vLLM's fused-expert"
         " loader", [(MXFP4, edit_config_class)]),
        ("quantization-name-registration",
         "gpt_oss_mxfp4 in the platform allowlist, the quant_methods"
         " constants and the config map", [
             (QUANT_METHODS, edit_quant_name_constant),
             (PLATFORM, edit_platform_allowlist),
             (QUANTIZATION, lambda text: edit_config_map(text, mxfp4()))
         ]),
        ("moe-forward-input-ids",
         "input_ids on the MoE forward, which vLLM passes", [
             (MXFP4, edit_forward_input_ids)
         ]),
        ("expert-padding-128",
         "zero-pad expert weights and biases to multiples of 128 (gmm_v2)", [
             (MXFP4, edit_expert_padding)
         ]),
        ("unloaded-moe-tail",
         "zero the MoE-TP intermediate tail the gpt-oss loader never writes", [
             (MXFP4, edit_unloaded_tail)
         ]),
        ("hd64-attention-sinks",
         "head_dim 64 layers stay on the hd64 kernel, which has attention"
         " sinks", [(ATTENTION, edit_hd64_attention_sinks)]),
    ]


def plan(sources):
    """Applies every fix to `sources` in memory.

    Returns (log lines, failures). Nothing is written here.
    """
    log, failures = [], []
    for name, what, edits in fixes(sources):
        staged = {}  # path -> text, kept only if every edit of the fix holds
        notes = []
        try:
            for candidates, edit in edits:
                path = sources.locate(candidates)
                before = staged.get(path, sources.read(path))
                try:
                    staged[path] = edit(before)
                    notes.append(("applied", rel(sources, path)))
                except Present as e:
                    notes.append((f"present, {e}", rel(sources, path)))
        except Mismatch as e:
            failures.append(f"{name} ({what}): {e}")
            continue
        for path, text in staged.items():
            try:
                ast.parse(text)
            except SyntaxError as e:
                failures.append(f"{name}: the patched {rel(sources, path)}"
                                f" would not parse: {e.msg} at line"
                                f" {e.lineno}")
                break
        else:
            sources.text.update(staged)
            for state, path in notes:
                log.append(f"  {name}: {state}: {path}")
    return log, failures


def main(argv):
    if len(argv) > 2:
        print(f"usage: {PATCH_NAME} [<plugin source root>]", file=sys.stderr)
        return 2
    root = argv[1] if len(argv) == 2 else find_root()
    if not root:
        print(
            f"vllm_torchtpu is not importable with {sys.executable};"
            " cannot patch it",
            file=sys.stderr)
        return 3
    root = os.path.abspath(root)
    if not os.path.isdir(os.path.join(root, "vllm_torchtpu")):
        print(f"no vllm_torchtpu/ under {root}", file=sys.stderr)
        return 3
    sources = Sources(root)
    log, failures = plan(sources)
    if failures:
        print(f"vllm patch does not apply: {PATCH_NAME} to {root}.",
              file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print(
            "Nothing was changed. The plugin revision moved an anchor these"
            " fixes rely on; check whether this image already fixes the"
            " problem before re-basing the fix.",
            file=sys.stderr)
        return 4
    print("\n".join(log))
    if sources.changed():
        sources.write()
        print(f"vllm patch applied: {PATCH_NAME} ({root})")
    else:
        print(f"vllm patch already applied: {PATCH_NAME} ({root})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
