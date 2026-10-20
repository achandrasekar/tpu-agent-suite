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
"""Locks the donation contract of `compile_and_align`.

Getting it wrong fails one of three ways, none visible in the output: an
undonated in-place kernel is timed together with a full copy of its buffer, a
donated buffer is reused after XLA consumed it, or every timed kernel pays a
compile that finds nothing to donate.
"""

import os
import types
import unittest

COMMON_PATH = os.path.join(os.path.dirname(__file__), "kernel_exec_common.py")

HEADER = ("HloModule jit_wrapped, is_scheduled=true, "
          "input_output_alias={{ {} }}, "
          "entry_computation_layout={{(f32[8]{{0}})->f32[8]{{0}}}}")


class _Array:
    """A stand-in device buffer of `nbytes` bytes; `name` says where it came from."""

    dtype = types.SimpleNamespace(itemsize=1, kind="f")

    def __init__(self, name, nbytes, shape=None, dtype=None):
        self.name = name
        self.shape = shape or (nbytes, )
        self.dtype = dtype or _Array.dtype
        self.deleted = False

    def is_deleted(self):
        return self.deleted


class _HostArray:
    """A stand-in numpy array: array-like, but not a device buffer."""

    dtype = _Array.dtype

    def __init__(self, name, nbytes):
        self.name = name
        self.shape = (nbytes, )

    def __array__(self):
        raise AssertionError("never read on the host")


INT8 = types.SimpleNamespace(itemsize=1, kind="i")


def _like(name, arr):
    return _Array(name, 0, shape=arr.shape, dtype=arr.dtype)


def _make_jax(alias_for, consumes_every_donation=False, copies_for=None):
    """A jax stand-in whose executables alias what `alias_for(donate)` says.

    A call marks the aliased inputs consumed -- or, with
    `consumes_every_donation`, every donated input, which is the jax behaviour
    `compile_and_align` has to detect rather than assume away.
    """
    jax = types.SimpleNamespace(Array=_Array, CALLS=[], COMPILES=[])

    class Compiled:

        def __init__(self, fn, donate):
            self.fn, self.donate = fn, donate

        def as_text(self):
            copies = copies_for(self.donate) if copies_for else ()
            # Copied aside, unchanged -- plus the kernel's own re-layout of the
            # same input, which `_copied_params` must not count.
            entry = "".join(f"  %p{i} = f32[8]{{0}} parameter({i})\n"
                            f"  %copy.{i} = f32[8]{{0}} copy(%p{i})\n"
                            f"  %bc{i} = f32[2,4]{{1,0}} bitcast(%p{i})\n"
                            f"  %t{i} = f32[2,4]{{0,1}} copy(%bc{i})\n"
                            for i in copies)
            return (HEADER.format(alias_for(self.donate)) +
                    "\nENTRY %main {\n" + entry + "}\n")

        def __call__(self, *xs):
            jax.CALLS.append(tuple(x.name for x in xs))
            aliased = ns_alias_map(self)
            for i in self.donate:
                if consumes_every_donation or i in aliased:
                    xs[i].deleted = True
            return self.fn(*xs)

    def jit(fn, donate_argnums=(), keep_unused=False, **kwargs):
        lowered = types.SimpleNamespace()
        lowered.compile = lambda: (jax.COMPILES.append(donate_argnums) or
                                   Compiled(fn, donate_argnums))
        # Called directly (the setup copies), it is not recorded: only calls
        # of the compiled executable go in CALLS.
        return type(
            "Jitted", (), {
                "lower": staticmethod(lambda *xs: lowered),
                "__call__": staticmethod(fn),
            })()

    jax.jit = jit
    jax.eval_shape = lambda fn, *xs: fn(*xs)
    jax.numpy = types.SimpleNamespace(
        floating="f", issubdtype=lambda dtype, kind: dtype.kind == kind)
    jax.block_until_ready = lambda x: x
    jax.device_put = lambda x: _like(f"device({x.name})", x)
    jax.lax = types.SimpleNamespace(optimization_barrier=lambda xs: tuple(
        _like(f"copy({x.name})", x) for x in xs))
    jax.tree_util = types.SimpleNamespace(
        tree_flatten=lambda t: (list(t), None),
        tree_unflatten=lambda _, leaves: tuple(leaves),
        tree_leaves=lambda t: list(t) if isinstance(t, tuple) else [t])
    ns_alias_map = _load(jax)["_alias_map"]
    return jax


def _load(jax):
    namespace = {"jax": jax}
    with open(COMMON_PATH) as handle:
        exec(compile(handle.read(), COMMON_PATH, "exec"), namespace)  # pylint: disable=exec-used
    return namespace


def _cache_update(cache, update, scale):
    """Returns a new cache (output 0, aliasable) and a 4-byte summary (output 1)."""
    return (_like(f"out({cache.name})", cache), _Array(f"sum({update.name})",
                                                       4))


def _add(x, y):
    """Two inputs the size of the output; XLA can alias only one of them."""
    return _like(f"out({x.name})", x)


ALIAS_INPUT_0 = "{0}: (0, {}, may-alias)"


class TestAliasMap(unittest.TestCase):

    def setUp(self):
        self.alias_map = _load(_make_jax(lambda _: ""))["_alias_map"]

    def _parse(self, entries):
        compiled = types.SimpleNamespace(
            as_text=lambda: HEADER.format(entries) + "\nENTRY ...")
        return self.alias_map(compiled)

    def test_tuple_outputs(self):
        self.assertEqual(
            self._parse("{0}: (1, {}, may-alias), {2}: (0, {}, must-alias)"), {
                1: 0,
                0: 2
            })

    def test_single_output(self):
        self.assertEqual(self._parse("{}: (2, {}, may-alias)"), {2: 0})

    def test_no_alias_attribute(self):
        compiled = types.SimpleNamespace(
            as_text=lambda: "HloModule jit_wrapped, is_scheduled=true\n")
        self.assertEqual(self.alias_map(compiled), {})

    def test_unreadable_executable_means_no_donation(self):
        self.assertEqual(self.alias_map(object()), {})


class TestCompileAndAlign(unittest.TestCase):

    def _run(self, jax, fn, args, static_argnums=(), calls=1):
        step = _load(jax)["compile_and_align"](fn, args, static_argnums)
        for _ in range(calls):
            step()

    def test_aliased_buffer_is_donated_and_threaded(self):
        """One compile, offering only the cache; each call gets the last output."""
        jax = _make_jax(lambda donate: ALIAS_INPUT_0 if 0 in donate else "")
        self._run(jax,
                  _cache_update,
                  (_Array("cache", 1024), _Array("update", 64), 2.0), (2, ),
                  calls=3)

        self.assertEqual(jax.COMPILES, [(0, )],
                         "only the cache is the size of an output")
        self.assertEqual(jax.CALLS, [
            ("copy(cache)", "update"),
            ("out(copy(cache))", "update"),
            ("out(out(copy(cache)))", "update"),
        ])

    def test_no_input_the_size_of_an_output_compiles_once_undonated(self):
        """A matmul has nothing to alias, so it pays exactly the old compile."""
        jax = _make_jax(lambda donate: self.fail("nothing should be offered")
                        if donate else "")
        matmul = lambda a, b: _Array("out", 15)  # noqa: E731
        self._run(jax, matmul, (_Array("a", 6), _Array("b", 10)), calls=2)
        self.assertEqual(jax.COMPILES, [()])
        self.assertEqual(jax.CALLS, [("a", "b")] * 2)

    def test_reshaped_output_still_counts_as_aliasable(self):
        """Same bytes, different shape: XLA can still write it in place."""
        jax = _make_jax(lambda donate: ALIAS_INPUT_0 if 0 in donate else "")
        reshape = lambda cache: _Array("out", 0, shape=(1024, ))  # noqa: E731
        self._run(jax, reshape, (_Array("cache", 0, shape=(32, 32)), ))
        self.assertEqual(jax.COMPILES, [(0, )])

    def test_integer_input_is_never_offered(self):
        """XLA would write another output into it: lengths become garbage."""
        jax = _make_jax(lambda donate: self.fail("nothing should be offered")
                        if donate else "")
        sampler = lambda kv_lens: _like("tokens", kv_lens)  # noqa: E731
        self._run(jax, sampler, (_Array("kv_lens", 64, dtype=INT8), ), calls=2)
        self.assertEqual(jax.COMPILES, [()])
        self.assertEqual(jax.CALLS, [("kv_lens", )] * 2)

    def test_donation_answered_with_a_copy_is_withdrawn(self):
        """Flash attention: donating q made XLA copy q aside, so q is dropped."""
        jax = _make_jax(lambda donate: ALIAS_INPUT_0 if 0 in donate else "",
                        copies_for=lambda donate: {0} & set(donate))
        self._run(jax, _add, (_Array("q", 8), _Array("k", 8)), calls=2)
        self.assertEqual(jax.COMPILES, [(0, 1), (1, )])
        self.assertTrue(all(call[0] == "q" for call in jax.CALLS), jax.CALLS)

    def test_kernels_own_relayout_copy_is_not_a_donation_copy(self):
        """rpav3 reshapes and transposes queries itself, donated or not."""
        jax = _make_jax(lambda donate: ALIAS_INPUT_0 if 0 in donate else "")
        compiled = types.SimpleNamespace(as_text=lambda: (
            "HloModule m\nENTRY %main {\n"
            "  %q = bf16[64,32,128]{2,1,0} parameter(0)\n"
            "  %bitcast.8 = bf16[64,8,4,128]{3,2,1,0} bitcast(%q)\n"
            "  %copy.7 = bf16[64,8,4,128]{3,1,2,0} copy(%bitcast.8)\n"
            "  %copy.9 = bf16[64,32,128]{2,0,1} copy(%q)\n"
            "}\n"))
        self.assertEqual(_load(jax)["_copied_params"](compiled), set())

    def test_copies_that_persist_mean_no_donation(self):
        """Dropping q makes XLA copy k instead: donate nothing at all."""
        jax = _make_jax(lambda donate: ALIAS_INPUT_0 if 0 in donate else "",
                        copies_for=lambda donate: {0}
                        if 0 in donate else set(donate))
        self._run(jax, _add, (_Array("q", 8), _Array("k", 8)))
        self.assertEqual(jax.COMPILES, [(0, 1), (1, ), ()])
        self.assertEqual(jax.CALLS, [("q", "k")])

    def test_unaliased_offer_that_survives_keeps_one_compile(self):
        """jax 0.11 leaves an unaliased donation untouched: no recompile."""
        jax = _make_jax(lambda donate: ALIAS_INPUT_0)
        self._run(jax, _add, (_Array("x", 8), _Array("y", 8)), calls=2)
        self.assertEqual(jax.COMPILES, [(0, 1)])
        # The first call is the check that `y` survived, made before timing.
        self.assertEqual(jax.CALLS, [
            ("copy(x)", "copy(y)"),
            ("out(copy(x))", "copy(y)"),
            ("out(out(copy(x)))", "copy(y)"),
        ])

    def test_unaliased_offer_that_is_consumed_triggers_a_recompile(self):
        """A jax that consumes every donation would free `y` before call two."""
        jax = _make_jax(lambda donate: ALIAS_INPUT_0,
                        consumes_every_donation=True)
        self._run(jax, _add, (_Array("x", 8), _Array("y", 8)), calls=2)
        self.assertEqual(jax.COMPILES, [(0, 1), (0, )])
        self.assertEqual(jax.CALLS, [
            ("copy(x)", "copy(y)"),
            ("copy(x)", "y"),
            ("out(copy(x))", "y"),
        ])

    def test_unstable_aliasing_falls_back_to_no_donation(self):
        """The recompile aliases nothing, so donating `x` would free it."""
        jax = _make_jax(lambda donate: ALIAS_INPUT_0
                        if len(donate) == 2 else "",
                        consumes_every_donation=True)
        self._run(jax, _add, (_Array("x", 8), _Array("y", 8)))
        self.assertEqual(jax.COMPILES, [(0, 1), (0, ), ()])
        self.assertEqual(jax.CALLS, [("copy(x)", "copy(y)"), ("x", "y")])

    def test_untraceable_kernel_offers_every_array(self):
        """No shapes to compare: offer every float, as before the size check."""
        jax = _make_jax(lambda donate: ALIAS_INPUT_0 if 0 in donate else "")

        def no_eval_shape(*_):
            raise RuntimeError("cannot trace")

        jax.eval_shape = no_eval_shape
        self._run(jax, _cache_update,
                  (_Array("cache", 1024), _Array("update", 64), 2.0), (2, ))
        self.assertEqual(jax.COMPILES[0], (0, 1))

    def test_host_array_is_uploaded_once_and_donated(self):
        """A numpy cache cannot be donated, so it would keep the copy."""
        jax = _make_jax(lambda donate: ALIAS_INPUT_0 if 0 in donate else "")
        self._run(jax,
                  _cache_update,
                  (_HostArray("cache", 1024), _Array("update", 64), 2.0),
                  (2, ),
                  calls=2)
        self.assertEqual(jax.COMPILES, [(0, )])
        self.assertEqual(jax.CALLS, [
            ("copy(device(cache))", "update"),
            ("out(copy(device(cache)))", "update"),
        ])


if __name__ == "__main__":
    unittest.main()
