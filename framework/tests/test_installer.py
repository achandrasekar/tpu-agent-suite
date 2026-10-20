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
"""Unit tests for installer module (install, upgrade, rollback, uninstall)."""

import contextlib
import functools
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from unittest import mock

from framework import compiler, installer, utils

Path = pathlib.Path


def profile_env(backend: str, **params: Any) -> dict[str, Any]:
    """An environment.json holding one profile, `default`, the default."""
    profile = {"backend": backend, **({"params": params} if params else {})}
    return {"default": "default", "profiles": {"default": profile}}


class TestInstaller(unittest.TestCase):
    """Unit tests for installer functions, pre-validations, collisions, and rollback."""

    def setUp(self):
        super().setUp()
        # The host check shells out to python3; its own test is in
        # TestHostCheck.
        host_check = mock.patch.object(installer, "check_host")
        self.check_host = host_check.start()
        self.addCleanup(host_check.stop)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        self.central_home.mkdir()
        self.old_tpu_agent_suite_home = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = str(self.central_home)

        self.pkg_dir = self._create_test_package()
        self.dist_dir = self.temp_dir / "dist"
        compiler.translate(self.pkg_dir, "claude-code", self.dist_dir)

    def tearDown(self):
        if self.old_tpu_agent_suite_home is not None:
            os.environ["TPU_AGENT_SUITE_HOME"] = self.old_tpu_agent_suite_home
        else:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _create_test_package(self) -> Path:
        pkg_dir = self.temp_dir / "pkg"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "name":
            "pkg",
            "version":
            "1.0.0",
            "description":
            "Installer test package",
            "skills": ["entry"],
            "copy": ["schemas", "environment"],
            "environment": {
                "questions": "environment/questions.json",
                "schema": "schemas/env.json",
            },
            "compatibility_targets": [{
                "name": "claude",
                "harness": "claude-code",
                "versions": ">=2.0.0",
            }],
        }
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        (pkg_dir / "skills" / "entry").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n"
            "delegates:\n"
            "  - helper\n"
            "---\n"
            "Call {{agent:helper}}.",
            encoding="utf-8")
        (pkg_dir / "agents" / "helper.md").write_text(
            "---\n"
            "name: helper\n"
            "description: Helper\n"
            "tools:\n"
            "  - read\n"
            "delegates: []\n"
            "accepts: schemas/in.json\n"
            "produces: schemas/out.json\n"
            "---\n"
            "Help.",
            encoding="utf-8")
        (pkg_dir / "schemas").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "schemas" / "in.json").write_text(json.dumps(
            {"type": "object"}),
                                                     encoding="utf-8")
        (pkg_dir / "schemas" / "out.json").write_text(json.dumps(
            {"type": "object"}),
                                                      encoding="utf-8")
        (pkg_dir / "schemas" / "env.json").write_text(
            json.dumps({
                "type": "object",
                "properties": {
                    "tpu_target": {
                        "type": "string"
                    },
                    "timeout": {
                        "type": "integer"
                    },
                    # What every install writes; real schemas say the same.
                    "revision": {
                        "type": "integer"
                    },
                    "default": {
                        "type": ["string", "null"]
                    },
                    "profiles": {
                        "type": "object"
                    },
                },
            }),
            encoding="utf-8",
        )
        (pkg_dir / "environment").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "environment" / "questions.json").write_text(
            json.dumps({
                "questions": [
                    {
                        "name": "tpu_target",
                        "prompt": "TPU target",
                        "type": "string",
                        "default": "local-tpu",
                    },
                    {
                        "name": "timeout",
                        "prompt": "Timeout",
                        "type": "integer",
                        "default": 30,
                    },
                ]
            }),
            encoding="utf-8",
        )
        return pkg_dir

    def test_install_and_environment_resolution(self):
        workspace = self.temp_dir / "workspace"
        workspace.mkdir()

        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            non_interactive=True,
            upgrade=False,
            skip_harness_check=True,
        )

        env_file = workspace / ".tpu-agent-suite" / "environment.json"
        self.assertTrue(env_file.is_file())
        env = json.loads(env_file.read_text(encoding="utf-8"))
        self.assertEqual(env["revision"], 1)

        ownership_file = workspace / ".tpu-agent-suite" / "ownership.json"
        self.assertTrue(ownership_file.is_file())

    def test_a_fresh_install_asks_nothing_and_leaves_compute_to_the_session(
            self):
        workspace = self.temp_dir / "workspace_fresh"
        workspace.mkdir()

        def main(*flags):
            out = io.StringIO()
            with mock.patch.object(installer.sys.stdin, "isatty",
                                   return_value=True), \
                    mock.patch("builtins.input",
                               side_effect=AssertionError("asked")), \
                    contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(io.StringIO()):
                rc = installer.main([
                    "install",
                    str(self.dist_dir), "--destination",
                    str(workspace), "--skip-harness-check",
                    "--relocated-store", *flags
                ])
            return rc, out.getvalue()

        rc, out = main()
        self.assertEqual(rc, 0)
        env_file = workspace / ".tpu-agent-suite" / "environment.json"
        self.assertEqual(json.loads(env_file.read_text()), {
            "revision": 1,
            "default": None,
            "profiles": {}
        })
        self.assertIn("The first session sets it up", out)
        self.check_host.assert_called_once()
        self.assertEqual(main("--skip-host-check")[0], 0)
        self.check_host.assert_called_once()

    def test_install_places_every_shipped_agent(self):
        """A normal install must leave each shipped agent somewhere loadable."""
        workspace = self.temp_dir / "workspace_agents"
        workspace.mkdir()
        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
        )
        self.assertTrue(
            (workspace / ".claude" / "agents" / "helper.md").is_file())

    def test_a_distribution_with_no_agents_is_not_a_reachability_problem(self):
        """Nothing shipped means nothing to be unreachable; stay silent."""
        empty = self.temp_dir / "skills_only_dist"
        (empty / "skills").mkdir(parents=True)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            installer.verify_agents_reachable(empty, [])
        self.assertEqual(err.getvalue(), "")

    def test_warns_when_workspace_has_no_repository_root(self):
        """Workspace discovery stops at a repository root, so agents installed
        outside one are present and invisible."""
        workspace = self.temp_dir / "no_repo_workspace"
        (workspace / ".agents" / "agents").mkdir(parents=True)
        copies = [(Path("dist/agents/helper.md"),
                   workspace / ".agents" / "agents" / "helper.md")]

        with contextlib.redirect_stderr(io.StringIO()) as err:
            installer.warn_if_workspace_not_discoverable(workspace, copies)
        self.assertIn("not inside a git repository", err.getvalue())

        (workspace / ".git").mkdir()
        with contextlib.redirect_stderr(io.StringIO()) as err:
            installer.warn_if_workspace_not_discoverable(workspace, copies)
        self.assertEqual(err.getvalue(), "")

    def test_no_agents_means_no_repository_warning(self):
        """A package that installs no agents has nothing to be undiscoverable."""
        workspace = self.temp_dir / "skills_only_workspace"
        workspace.mkdir()
        copies = [(Path("dist/skills/entry/SKILL.md"),
                   workspace / ".agents" / "skills" / "entry" / "SKILL.md")]
        with contextlib.redirect_stderr(io.StringIO()) as err:
            installer.warn_if_workspace_not_discoverable(workspace, copies)
        self.assertEqual(err.getvalue(), "")

    def test_install_without_a_terminal_fails_fast_naming_headless_flags(self):
        """Without a TTY an interactive install would hang on (or die
        mid-prompt reading) stdin; it must refuse before doing anything.
        --force never asks, so it installs."""
        argv = [
            "install",
            str(self.temp_dir), "--destination",
            str(self.temp_dir)
        ]
        with mock.patch.object(installer.sys.stdin, "isatty",
                               return_value=False), \
                mock.patch.object(installer, "install") as install, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            ret = installer.main(argv)
            self.assertEqual(ret, 1)
            install.assert_not_called()
            self.assertIn("--non-interactive", err.getvalue())
            installer.main(argv + ["--force", "--relocated-store"])
            install.assert_called_once()

    def test_install_refuses_a_relocated_store(self):
        """Hooks always run ~/.tpu-agent-suite/.../runtime.py,
        so a store TPU_AGENT_SUITE_HOME moved elsewhere breaks every hook.
        Refuse before writing anything."""
        workspace = self.temp_dir / "workspace_relocated"
        workspace.mkdir()
        argv = [
            "install",
            str(self.dist_dir), "--destination",
            str(workspace), "--non-interactive", "--skip-harness-check"
        ]
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(installer.main(argv), 1)
        self.assertIn("TPU_AGENT_SUITE_HOME", err.getvalue())
        self.assertEqual(list(workspace.iterdir()), [])
        self.assertEqual(list(self.central_home.iterdir()), [])

        # The store at its default place is not relocated, even when the
        # variable names it (as the eval containers do).
        fake_home = self.temp_dir / "fake_home"
        with mock.patch.object(Path, "home", return_value=fake_home), \
                mock.patch.dict(os.environ, {"TPU_AGENT_SUITE_HOME": str(
                    fake_home / utils.TPU_AGENT_SUITE_DIRNAME)}):
            installer.check_store_not_relocated()

    def test_install_rejects_a_plan_that_drops_agents(self):
        """Shipping agents that the plan never installs must fail loudly.

        Left alone it succeeds silently, and the damage surfaces much later as
        an orchestrator that was told to delegate, found nothing registered,
        and ran the agent's instructions inline instead.
        """
        plan_path = self.dist_dir / "tpu-agent-suite-install.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["copies"] = [
            c for c in plan["copies"] if "agents" not in c["destination"]
        ]
        plan_path.write_text(json.dumps(plan), encoding="utf-8")

        workspace = self.temp_dir / "workspace_dropped_agents"
        workspace.mkdir()
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            installer.install(
                distribution=self.dist_dir,
                destination=workspace,
                non_interactive=True,
                skip_harness_check=True,
            )
        self.assertIn("never installs", str(ctx.exception))
        self.assertIn("helper", str(ctx.exception))

    def _dist_with_real_questionnaire(self, name: str) -> Path:
        """A build of the test package with the shipped questions and schema."""
        real_env = (pathlib.Path(__file__).resolve().parents[2] / "package" /
                    "environment")
        real_schema = real_env.parent / "schemas" / "environment.json"
        if not real_env.is_dir() or not real_schema.is_file():
            self.skipTest("package questionnaire not available in this layout")
        shutil.copy2(real_env / "questions.json",
                     self.pkg_dir / "environment" / "questions.json")
        shutil.copy2(real_schema, self.pkg_dir / "schemas" / "env.json")
        dist = self.temp_dir / name
        compiler.translate(self.pkg_dir, "claude-code", dist)
        return dist

    def test_a_headless_profile_reaches_the_compute_driver(self):
        """A profile saved after install is one the runtime resolves.

        Headless installs (CI, evals) install, then `compute setup apply`.
        This drives the shipped questions.json and schema through both and
        resolves a driver from the result, so a mismatch fails here rather
        than on the first `compute run`.
        """
        from framework.compute import config as compute_config
        from framework.compute import setup as compute_setup

        dist = self._dist_with_real_questionnaire("dist_real_env")
        workspace = self.temp_dir / "workspace_env"
        workspace.mkdir()
        installer.install(distribution=dist,
                          destination=workspace,
                          non_interactive=True,
                          skip_harness_check=True)
        result = compute_setup.apply_setup(profile="default",
                                           answers={
                                               "backend":
                                               "gce_tpu_vm",
                                               "tpu_name":
                                               "node-1",
                                               "zone":
                                               "us-central2-b",
                                               "project":
                                               "proj",
                                               "python_interpreter":
                                               "/opt/venv/bin/python",
                                           },
                                           package="pkg",
                                           workspace=workspace)
        self.assertEqual(result["status"], "ok", result)

        driver = compute_config.resolve_driver(workspace=workspace,
                                               package="pkg")
        self.assertEqual(driver.tpu_name, "node-1")
        self.assertEqual(driver.python_interpreter, "/opt/venv/bin/python")

    def test_collision_detection(self):
        workspace = self.temp_dir / "workspace_collision"
        workspace.mkdir()
        (workspace / ".claude" / "skills" / "entry").mkdir(parents=True)
        (workspace / ".claude" / "skills" / "entry" / "SKILL.md").write_text(
            "EXISTING UNMANAGED FILE", encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            installer.install(
                distribution=self.dist_dir,
                destination=workspace,
                non_interactive=True,
                skip_harness_check=True,
            )
        self.assertIn("refusing to overwrite", str(ctx.exception).lower())

        # Test --force bypasses collision check and installs
        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            non_interactive=True,
            force=True,
            skip_harness_check=True,
        )
        self.assertTrue((workspace / ".claude" / "skills" / "entry" /
                         "SKILL.md").is_file())

    def test_upgrade_and_force_modified_files(self):
        workspace = self.temp_dir / "workspace_upgrade"
        workspace.mkdir()

        # Initial install
        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
        )

        # Modify managed file
        agent_file = workspace / ".claude" / "agents" / "helper.md"
        agent_file.write_text("# USER MODIFIED AGENT", encoding="utf-8")

        # Upgrade without force should fail in non-interactive
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            installer.install(
                distribution=self.dist_dir,
                destination=workspace,
                non_interactive=True,
                skip_harness_check=True,
            )
        self.assertIn("managed content changed", str(ctx.exception).lower())

        # Upgrade with force should succeed
        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            non_interactive=True,
            force=True,
            skip_harness_check=True,
        )
        self.assertNotIn("# USER MODIFIED AGENT",
                         agent_file.read_text(encoding="utf-8"))

    def test_reinstall_keeps_profiles_written_by_config(self):
        workspace = self.temp_dir / "workspace_profiles"
        workspace.mkdir()
        install = functools.partial(installer.install,
                                    distribution=self.dist_dir,
                                    destination=workspace,
                                    non_interactive=True,
                                    skip_harness_check=True)
        install()
        plan = json.loads(
            (self.dist_dir / "tpu-agent-suite-install.json").read_text())
        env_path = workspace / plan["environment_path"]
        profiles = {
            "revision": 5,
            "default": "vm",
            "profiles": {
                "vm": {
                    "backend": "local",
                    "params": {}
                }
            }
        }
        env_path.write_text(json.dumps(profiles), encoding="utf-8")
        results = env_path.with_name("verification.json")
        results.write_text('{"h": {"profile": "vm"}}')
        for kwargs in ({}, {"force": True}, {"clean": True}):
            with self.subTest(**kwargs):
                install(**kwargs)
                self.assertEqual(json.loads(env_path.read_text()), profiles)
                self.assertEqual(json.loads(results.read_text()),
                                 {"h": {
                                     "profile": "vm"
                                 }})
        # verify's results go with the package.
        installer.uninstall(destination=workspace, package="pkg")
        self.assertFalse(results.parent.exists())

    def test_uninstall_removes_the_environment_compute_setup_wrote(self):
        """Left behind, it would serve any package that looked there."""
        workspace = self.temp_dir / "workspace_setup_then_uninstall"
        workspace.mkdir()
        installer.install(distribution=self.dist_dir,
                          destination=workspace,
                          non_interactive=True,
                          skip_harness_check=True)
        env_path = workspace / ".tpu-agent-suite" / "environment.json"
        env_path.write_text('{"revision": 2, "default": "vm", "profiles": {}}',
                            encoding="utf-8")

        installer.uninstall(destination=workspace)
        self.assertFalse(env_path.parent.exists())
        self.assertEqual(utils.resolve_environment("other", workspace),
                         ({}, None))

    def test_a_project_install_in_home_goes_where_uninstall_looks(self):
        """In $HOME, .tpu-agent-suite/ is the central store, wherever it is."""
        home = self.temp_dir / "home"
        home.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            installer.install(distribution=self.dist_dir,
                              destination=home,
                              scope="project",
                              non_interactive=True,
                              skip_harness_check=True)
            self.assertTrue((self.central_home / "ownership.json").is_file())
            self.assertTrue((self.central_home / "environment.json").is_file())
            self.assertFalse((home / ".tpu-agent-suite").exists())
            installer.uninstall(destination=home)
        self.assertFalse((self.central_home / "ownership.json").exists())

    def test_a_reinstall_leaves_a_broken_environment_json_as_it_is(self):
        """The runtime reports it; the installer does not read it."""
        workspace = self.temp_dir / "workspace_malformed"
        workspace.mkdir()
        install = functools.partial(installer.install,
                                    distribution=self.dist_dir,
                                    destination=workspace,
                                    non_interactive=True,
                                    skip_harness_check=True)
        install()
        env_file = workspace / ".tpu-agent-suite" / "environment.json"
        env_file.write_text("not json")
        install(force=True)
        self.assertEqual(env_file.read_text(), "not json")

    def test_user_scope_keeps_one_environment_json_in_the_central_store(self):
        from framework.compute import setup as compute_setup

        home = self.temp_dir / "home"
        central = self.central_home / "environment.json"
        central.parent.mkdir(parents=True, exist_ok=True)
        saved = {
            "revision": 2,
            "default": "a",
            "profiles": {
                "a": {
                    "backend": "local"
                }
            }
        }
        central.write_text(json.dumps(saved))
        install = functools.partial(installer.install,
                                    scope="user",
                                    non_interactive=True,
                                    skip_harness_check=True)
        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            install(distribution=self.dist_dir)
            env, source = utils.resolve_environment("pkg@1.0.0", home)
            # A bare name, as typed by hand in $HOME, finds the same file
            # rather than making a new copy beside it.
            self.assertEqual(compute_setup._environment("pkg", home),
                             (env, source))
        self.assertEqual((env, source.resolve()), (saved, central.resolve()))
        self.assertEqual(
            utils.load_json(self.central_home / "ownership.json")["package"],
            "pkg")
        self.assertFalse(
            (home / ".tpu-agent-suite" / "environment.json").exists())
        # The file sits outside every version's directory, so an upgrade
        # finds it where the last version left it.
        manifest = self.pkg_dir / "package.release.json"
        manifest.write_text(manifest.read_text().replace('"1.0.0"', '"2.0.0"'))
        dist2 = self.temp_dir / "dist2"
        compiler.translate(self.pkg_dir, "claude-code", dist2)
        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            install(distribution=dist2)
            env, source = utils.resolve_environment("pkg@2.0.0", home)
            self.assertEqual((env, source.resolve()),
                             (saved, central.resolve()))
            # A user-scope uninstall removes it: it is in no ledger.
            installer.uninstall(scope="user")
        self.assertFalse(central.exists())
        self.assertFalse((self.central_home / "ownership.json").exists())

    def _other_package_dist(self) -> Path:
        """A distribution of a second package, `other`."""
        other_pkg = self.temp_dir / "other_pkg"
        shutil.copytree(self.pkg_dir, other_pkg)
        manifest = json.loads((other_pkg / "package.release.json").read_text())
        manifest["name"] = "other"
        (other_pkg / "package.release.json").write_text(json.dumps(manifest))
        other_dist = self.temp_dir / "other_dist"
        compiler.translate(other_pkg, "claude-code", other_dist)
        return other_dist

    def test_a_workspace_holds_one_package_at_a_time(self):
        """A second package is refused; the installed one still upgrades."""
        workspace = self.temp_dir / "workspace_one_package"
        workspace.mkdir()
        install = functools.partial(installer.install,
                                    destination=workspace,
                                    non_interactive=True,
                                    skip_harness_check=True)
        install(distribution=self.dist_dir)

        with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                    "already has pkg installed"):
            install(distribution=self._other_package_dist())
        ledger = workspace / ".tpu-agent-suite" / "ownership.json"
        self.assertEqual(utils.load_json(ledger)["package"], "pkg")

        install(distribution=self.dist_dir)
        self.assertEqual(utils.load_json(ledger)["package"], "pkg")

    def test_uninstall_reads_the_package_from_the_ledger(self):
        """No --package needed; given, it has to match what is installed."""
        workspace = self.temp_dir / "workspace_ledger_uninstall"
        workspace.mkdir()
        installer.install(distribution=self.dist_dir,
                          destination=workspace,
                          non_interactive=True,
                          skip_harness_check=True)

        with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                    "has pkg installed, not other"):
            installer.uninstall(destination=workspace, package="other")
        installer.uninstall(destination=workspace)
        self.assertFalse(
            (workspace / ".tpu-agent-suite" / "ownership.json").exists())
        with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                    "nothing is installed"):
            installer.uninstall(destination=workspace)

    def test_the_installed_package_uninstalls_without_its_distribution(self):
        """The central copy of the installer removes the install by itself.

        Both installers run as programs, as a user runs them: the
        distribution's framework/installer.py, and the central one, run after
        the distribution is gone. Releasing the last reference deletes the
        directory that installer runs from.
        """
        self.assertFalse((self.dist_dir / "utils.py").exists())
        workspace = self.temp_dir / "workspace_central_uninstall"
        workspace.mkdir()

        def run(script: Path, *argv: str) -> None:
            result = subprocess.run([
                sys.executable,
                str(script), *argv, "--destination",
                str(workspace)
            ],
                                    capture_output=True,
                                    text=True,
                                    cwd=self.temp_dir,
                                    env={
                                        k: v
                                        for k, v in os.environ.items()
                                        if k != "PYTHONPATH"
                                    },
                                    check=False)
            self.assertEqual(result.returncode, 0, result.stderr)

        run(self.dist_dir / "framework" / "installer.py", "install",
            "--non-interactive", "--skip-harness-check", "--skip-host-check",
            "--relocated-store")
        central_installer = (utils.get_package_storage_root("pkg", "1.0.0") /
                             "framework" / "installer.py")
        self.assertTrue(central_installer.is_file())

        shutil.rmtree(self.dist_dir)
        # Beside a build file but no plan, it is not a distribution.
        reinstall = subprocess.run([
            sys.executable,
            str(central_installer), "install", "--non-interactive",
            "--destination",
            str(workspace)
        ],
                                   capture_output=True,
                                   text=True,
                                   check=False)
        self.assertEqual(reinstall.returncode, 1)
        self.assertIn("distribution directory is required", reinstall.stderr)
        run(central_installer, "uninstall")
        self.assertFalse(
            (workspace / ".tpu-agent-suite" / "ownership.json").exists())
        self.assertFalse(central_installer.parent.exists())

    def test_a_ledger_of_another_schema_is_left_to_its_own_installer(self):
        """Install and uninstall refuse it, and name the installer that can."""
        workspace = self.temp_dir / "workspace_ledger_schema"
        workspace.mkdir()
        install = functools.partial(installer.install,
                                    distribution=self.dist_dir,
                                    destination=workspace,
                                    non_interactive=True,
                                    skip_harness_check=True)
        install()
        ledger = workspace / ".tpu-agent-suite" / "ownership.json"
        self.assertEqual(
            utils.load_json(ledger)["ledger_schema"], installer.LEDGER_SCHEMA)
        utils.write_json(
            ledger, {
                **utils.load_json(ledger), "ledger_schema":
                installer.LEDGER_SCHEMA + 1
            })

        command = re.escape(
            f"python3 {utils.get_package_storage_root('pkg', '1.0.0')}"
            f"/framework/installer.py uninstall --destination {workspace.resolve()}"
        )
        with self.assertRaisesRegex(utils.TpuAgentSuiteError, command):
            install()
        with self.assertRaisesRegex(utils.TpuAgentSuiteError, command):
            installer.uninstall(destination=workspace)
        self.assertEqual(
            utils.load_json(ledger)["ledger_schema"],
            installer.LEDGER_SCHEMA + 1)
        self.assertTrue(
            (workspace / ".claude" / "skills" / "entry" / "SKILL.md").exists())

    def test_uninstall_hands_off_to_the_installed_versions_installer(self):
        """Another installer passes uninstall to the one that wrote the ledger."""
        workspace = self.temp_dir / "workspace_handoff"
        workspace.mkdir()
        installer.install(distribution=self.dist_dir,
                          destination=workspace,
                          non_interactive=True,
                          skip_harness_check=True)
        central_installer = (utils.get_package_storage_root("pkg", "1.0.0") /
                             "framework" / "installer.py")
        handed_off = central_installer.with_name("handed_off.json")
        central_installer.write_text(
            "import json, sys\n"
            f"open({str(handed_off)!r}, 'w').write(json.dumps(sys.argv[1:]))\n",
            encoding="utf-8")

        argv = ["uninstall", "--destination", str(workspace)]
        self.assertEqual(installer.main(argv), 0)
        self.assertEqual(utils.load_json(handed_off), argv)
        self.assertTrue(
            (workspace / ".tpu-agent-suite" / "ownership.json").exists())

    def test_a_packages_installer_uninstalls_only_that_package(self):
        """pkg's installer, run where other is installed, leaves other be."""
        workspace = self.temp_dir / "workspace_other_package"
        workspace.mkdir()
        installer.install(distribution=self._other_package_dist(),
                          destination=workspace,
                          non_interactive=True,
                          skip_harness_check=True)

        result = subprocess.run([
            sys.executable,
            str(self.dist_dir / "framework" / "installer.py"), "uninstall",
            "--destination",
            str(workspace)
        ],
                                capture_output=True,
                                text=True,
                                env={
                                    k: v
                                    for k, v in os.environ.items()
                                    if k != "PYTHONPATH"
                                },
                                check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("has other installed, not pkg", result.stderr)
        self.assertEqual(
            utils.load_json(workspace / ".tpu-agent-suite" /
                            "ownership.json")["package"], "other")

    def test_upgrade_releases_previous_versions_central_ref(self):
        workspace = self.temp_dir / "workspace_ref_upgrade"
        workspace.mkdir()

        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
        )
        old_central = self.central_home / "packages" / "pkg@1.0.0"
        self.assertTrue(old_central.is_dir())

        # Same-version reinstall keeps the reference.
        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
        )
        refs = json.loads(
            (old_central / ".refs.json").read_text(encoding="utf-8"))
        self.assertIn(str(workspace.resolve()), refs["workspaces"])

        # Upgrade to 2.0.0.
        manifest = json.loads(
            (self.pkg_dir /
             "package.release.json").read_text(encoding="utf-8"))
        manifest["version"] = "2.0.0"
        (self.pkg_dir / "package.release.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        dist_v2 = self.temp_dir / "dist_v2"
        compiler.translate(self.pkg_dir, "claude-code", dist_v2)
        installer.install(
            distribution=dist_v2,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
        )

        # The old version directory is garbage-collected (this workspace
        # was its only reference) and the new version records the workspace.
        self.assertFalse(old_central.exists())
        new_refs = json.loads((self.central_home / "packages" / "pkg@2.0.0" /
                               ".refs.json").read_text(encoding="utf-8"))
        self.assertIn(str(workspace.resolve()), new_refs["workspaces"])

    def _install_with_target(self, workspace: Path, tpu_target: str) -> None:
        """Install the test package into `workspace`, set to `tpu_target`."""
        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
        )
        utils.write_json(workspace / ".tpu-agent-suite" / "environment.json",
                         {"tpu_target": tpu_target})

    def test_second_workspace_install_is_isolated_from_the_first(self):
        """Installing into a second project never points it at the first.

        Both installs share one central payload; everything a harness loads
        from the second workspace must still be its own.
        """
        project_1 = self.temp_dir / "project_1"
        project_2 = self.temp_dir / "project_2"
        project_1.mkdir()
        project_2.mkdir()
        self._install_with_target(project_1, "tpu-one")
        self._install_with_target(project_2, "tpu-two")

        for workspace, target in ((project_1, "tpu-one"), (project_2,
                                                           "tpu-two")):
            env, source = utils.resolve_environment("pkg", workspace)
            self.assertEqual(env["tpu_target"], target)
            self.assertEqual(
                source,
                workspace.resolve() / ".tpu-agent-suite" / "environment.json")

        # No installed file in project_2 mentions project_1, under either
        # spelling of its path (macOS temp dirs resolve through /private).
        p1_spellings = {str(project_1), str(project_1.resolve())}
        for path in project_2.rglob("*"):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for spelling in p1_spellings:
                self.assertNotIn(spelling, text,
                                 f"{path} references project_1")

        # Both workspaces hold a claim on the shared payload, so removing
        # project_1 leaves it in place for project_2.
        central = self.central_home / "packages" / "pkg@1.0.0"
        refs = json.loads((central / ".refs.json").read_text(encoding="utf-8"))
        self.assertEqual(
            refs["workspaces"],
            sorted([str(project_1.resolve()),
                    str(project_2.resolve())]))

        installer.uninstall(project_1, "pkg")
        self.assertTrue(central.is_dir())
        refs = json.loads((central / ".refs.json").read_text(encoding="utf-8"))
        self.assertEqual(refs["workspaces"], [str(project_2.resolve())])
        env, _ = utils.resolve_environment("pkg", project_2)
        self.assertEqual(env["tpu_target"], "tpu-two")
        self.assertTrue(
            (project_2 / ".claude" / "agents" / "helper.md").is_file())

    def test_nested_workspace_install_resolves_its_own_config(self):
        """A project inside another installed project gets its own install."""
        outer = self.temp_dir / "project_1"
        inner = outer / "project_2"
        inner.mkdir(parents=True)
        self._install_with_target(outer, "tpu-outer")
        self._install_with_target(inner, "tpu-inner")

        env, source = utils.resolve_environment("pkg", inner)
        self.assertEqual(env["tpu_target"], "tpu-inner")
        self.assertEqual(
            source,
            inner.resolve() / ".tpu-agent-suite" / "environment.json")
        self.assertTrue((inner / ".claude" / "agents" / "helper.md").is_file())

        # The outer install is untouched by the inner one.
        env, _ = utils.resolve_environment("pkg", outer)
        self.assertEqual(env["tpu_target"], "tpu-outer")

    def test_clean_wipe_installation(self):
        workspace = self.temp_dir / "workspace_clean"
        workspace.mkdir()

        # Initial install
        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
        )

        # Clean install in non-interactive mode
        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            non_interactive=True,
            clean=True,
            skip_harness_check=True,
        )
        self.assertTrue(
            (workspace / ".tpu-agent-suite" / "ownership.json").is_file())

    def test_uninstall_and_rollback(self):
        workspace = self.temp_dir / "workspace_uninstall"
        workspace.mkdir()

        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
        )

        # Modify an agent file
        agent_file = workspace / ".claude" / "agents" / "helper.md"
        self.assertTrue(agent_file.is_file())
        agent_file.write_text("# USER MODIFIED", encoding="utf-8")

        modified = installer.uninstall(workspace, "pkg")
        self.assertIn(".claude/agents/helper.md", modified)
        self.assertTrue(agent_file.is_file())  # Preserved
        self.assertFalse(
            (workspace / ".tpu-agent-suite" / "ownership.json").exists())

    def test_project_scope_requires_destination(self):
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            installer.install(
                distribution=self.dist_dir,
                scope="project",
                destination=None,
                skip_harness_check=True,
            )
        self.assertIn("--destination is required when scope is 'project'",
                      str(ctx.exception))

    def test_user_scope_rejects_destination(self):
        workspace = self.temp_dir / "workspace_user_reject"
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            installer.install(
                distribution=self.dist_dir,
                destination=workspace,
                scope="user",
                skip_harness_check=True,
            )
        self.assertIn("Cannot specify --destination when --scope is 'user'",
                      str(ctx.exception))

    def test_unknown_scope_raises_error(self):
        workspace = self.temp_dir / "workspace_unknown"
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            installer.resolve_install_destination(
                self.dist_dir,
                scope="local",
                destination=workspace,
            )
        self.assertIn("unknown scope 'local'. Must be 'project' or 'user'",
                      str(ctx.exception))

    def test_scope_user_resolves_to_home(self):
        """User scope roots at home, not at a per-harness subdirectory.

        Every harness names its own dotted directory in the copy plan
        (`.claude/agents`, `.codex/agents`), so those paths are already
        home-relative. Rooting them at ~/.claude put the agents in
        ~/.claude/.claude/agents, which no harness reads.
        """
        dest, scope = installer.resolve_install_destination(self.dist_dir,
                                                            scope="user",
                                                            destination=None)
        self.assertEqual(scope, "user")
        self.assertEqual(dest, Path.home().resolve())

    def test_scope_user_copy_targets_are_not_double_nested(self):
        """The symptom the resolution above exists to prevent."""
        dest, _ = installer.resolve_install_destination(self.dist_dir,
                                                        scope="user",
                                                        destination=None)
        plan = json.loads(
            (self.dist_dir /
             "tpu-agent-suite-install.json").read_text(encoding="utf-8"))
        for copy in plan["copies"]:
            resolved = dest / copy["destination"]
            self.assertNotIn(".claude/.claude", str(resolved))
            self.assertNotIn(".codex/.codex", str(resolved))

    def test_scope_user_uninstall_matches_install(self):
        """Uninstall must look where install wrote, or it silently finds nothing."""
        install_dest, _ = installer.resolve_install_destination(
            self.dist_dir, scope="user", destination=None)
        uninstall_dest = installer.resolve_uninstall_destination(
            scope="user", destination=None)
        self.assertEqual(install_dest, uninstall_dest)

    def test_scope_project_in_dist_dir_raises_error(self):
        """Cannot install into distribution directory itself with scope project."""
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            installer.resolve_install_destination(self.dist_dir,
                                                  scope="project",
                                                  destination=self.dist_dir)
        self.assertIn("Cannot install into the distribution directory itself",
                      str(ctx.exception))

    def test_scope_project_git_exclude(self):
        repo_dir = self.temp_dir / "git_repo"
        repo_dir.mkdir(parents=True)
        git_dir = repo_dir / ".git"
        git_dir.mkdir(parents=True)
        (git_dir / "info").mkdir(parents=True)

        installer.install(
            distribution=self.dist_dir,
            destination=repo_dir,
            scope="project",
            non_interactive=True,
            skip_harness_check=True,
        )
        exclude_file = git_dir / "info" / "exclude"
        self.assertTrue(exclude_file.is_file())
        content = exclude_file.read_text(encoding="utf-8")
        self.assertIn(".tpu-agent-suite/", content.splitlines())
        self.assertNotIn(".tpu-agent-suite/pkg/", content)

    def test_install_agent_scripts_without_explicit_copy_entry(self):
        (self.pkg_dir / "scripts").mkdir(parents=True, exist_ok=True)
        (self.pkg_dir / "scripts" / "helper.py").write_text("print('helper')",
                                                            encoding="utf-8")
        helper_md = (self.pkg_dir / "agents" /
                     "helper.md").read_text(encoding="utf-8")
        helper_md = helper_md.replace(
            "tools:\n  - read\n",
            "tools:\n  - read\nscripts:\n  - scripts/helper.py\n")
        (self.pkg_dir / "agents" / "helper.md").write_text(helper_md,
                                                           encoding="utf-8")

        dist = self.temp_dir / "dist_scripts"
        compiler.translate(self.pkg_dir, "claude-code", dist)

        workspace = self.temp_dir / "ws_scripts"
        workspace.mkdir()
        installer.install(
            distribution=dist,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
        )
        pkg_tag = utils.format_package_tag("pkg", "1.0.0")
        self.assertTrue((self.central_home / "packages" / pkg_tag / "scripts" /
                         "helper.py").is_file())

    def test_install_agent_references_without_explicit_copy_entry(self):
        (self.pkg_dir / "references").mkdir(parents=True, exist_ok=True)
        (self.pkg_dir / "references" / "guide.md").write_text("# Guide",
                                                              encoding="utf-8")
        helper_md = (self.pkg_dir / "agents" /
                     "helper.md").read_text(encoding="utf-8")
        helper_md = helper_md.replace(
            "tools:\n  - read\n",
            "tools:\n  - read\nreferences:\n  - references/guide.md\n")
        (self.pkg_dir / "agents" / "helper.md").write_text(helper_md,
                                                           encoding="utf-8")

        dist = self.temp_dir / "dist_refs"
        compiler.translate(self.pkg_dir, "claude-code", dist)

        workspace = self.temp_dir / "ws_refs"
        workspace.mkdir()
        installer.install(
            distribution=dist,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
        )
        pkg_tag = utils.format_package_tag("pkg", "1.0.0")
        self.assertTrue((self.central_home / "packages" / pkg_tag /
                         "references" / "guide.md").is_file())

    def test_install_script_tools_without_explicit_copy_entry(self):
        (self.pkg_dir / "tools").mkdir(parents=True, exist_ok=True)
        (self.pkg_dir / "tools" / "run.py").write_text("print('run')",
                                                       encoding="utf-8")
        pkg_json = json.loads(
            (self.pkg_dir /
             "package.release.json").read_text(encoding="utf-8"))
        pkg_json["script_tools"] = [{
            "name": "run_tool",
            "path": "tools/run.py",
            "visibility": "internal",
            "action_level": {
                "effect": "execute",
                "scope": "host",
            },
        }]
        (self.pkg_dir / "package.release.json").write_text(
            json.dumps(pkg_json), encoding="utf-8")
        helper_md = (self.pkg_dir / "agents" /
                     "helper.md").read_text(encoding="utf-8")
        helper_md = helper_md.replace("tools:\n  - read\n",
                                      "tools:\n  - read\n  - run_tool\n")
        (self.pkg_dir / "agents" / "helper.md").write_text(helper_md,
                                                           encoding="utf-8")

        dist = self.temp_dir / "dist_tools"
        compiler.translate(self.pkg_dir, "claude-code", dist)

        workspace = self.temp_dir / "ws_tools"
        workspace.mkdir()
        installer.install(
            distribution=dist,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
        )
        pkg_tag = utils.format_package_tag("pkg", "1.0.0")
        self.assertTrue((self.central_home / "packages" / pkg_tag / "tools" /
                         "run.py").is_file())

    def _create_pkg_with_hooks(self) -> Path:
        pkg_dir = Path(tempfile.mkdtemp(prefix="pkg_hooks_",
                                        dir=self.temp_dir))
        (pkg_dir / "scripts").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "scripts" / "custom_guard.py").write_text(
            "print('guarded')", encoding="utf-8")

        manifest = {
            "name":
            "pkg-hooks",
            "version":
            "1.0.0",
            "description":
            "Package with hooks",
            "skills": ["entry"],
            "copy": ["schemas", "environment"],
            "hooks": [{
                "name": "cmd-guard",
                "event": "PreToolUse",
                "matchers": ["read"],
                "command": "python3 check_input.py",
                "args": ["--strict"],
                "timeout": 5,
                "visibility": "external",
            }, {
                "name": "custom-guard",
                "event": "PreToolUse",
                "matchers": ["read"],
                "action": "script:scripts/custom_guard.py",
                "timeout": 10,
                "visibility": "external",
            }],
            "environment": {
                "questions": "environment/questions.json",
                "schema": "schemas/env.json",
            },
            "compatibility_targets": [
                {
                    "name": "claude",
                    "harness": "claude-code",
                    "versions": ">=2.0.0",
                },
                {
                    "name": "antigravity",
                    "harness": "antigravity",
                    "versions": ">=1.0.0",
                },
                {
                    "name": "codex",
                    "harness": "codex",
                    "versions": ">=0.145.0",
                },
            ],
        }
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        (pkg_dir / "skills" / "entry").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n"
            "delegates:\n"
            "  - helper\n"
            "---\n"
            "Call {{agent:helper}}.",
            encoding="utf-8")
        (pkg_dir / "agents" / "helper.md").write_text(
            "---\n"
            "name: helper\n"
            "description: Helper\n"
            "tools:\n"
            "  - read\n"
            "delegates: []\n"
            "accepts: schemas/in.json\n"
            "produces: schemas/out.json\n"
            "---\n"
            "Help.",
            encoding="utf-8")
        (pkg_dir / "schemas").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "schemas" / "in.json").write_text(json.dumps(
            {"type": "object"}),
                                                     encoding="utf-8")
        (pkg_dir / "schemas" / "out.json").write_text(json.dumps(
            {"type": "object"}),
                                                      encoding="utf-8")
        (pkg_dir / "schemas" / "env.json").write_text(json.dumps(
            {"type": "object"}),
                                                      encoding="utf-8")
        (pkg_dir / "environment").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "environment" / "questions.json").write_text(
            json.dumps({
                "questions": [{
                    "name": "target",
                    "prompt": "Target",
                    "type": "string",
                    "default": "local"
                }]
            }),
            encoding="utf-8",
        )
        return pkg_dir

    def test_install_and_uninstall_with_hooks(self):
        pkg_dir = self._create_pkg_with_hooks()
        dist = self.temp_dir / "dist_hooks"
        compiler.translate(pkg_dir, "claude-code", dist)
        self.assertTrue((dist / "project" / "hooks.json").is_file())
        self.assertTrue((dist / "scripts" / "custom_guard.py").is_file())

        workspace = self.temp_dir / "workspace_hooks"
        workspace.mkdir(parents=True, exist_ok=True)

        # 1. Install
        installer.install(
            distribution=dist,
            destination=workspace,
            scope=None,
            non_interactive=True,
            skip_harness_check=True,
        )

        # Verify exact installed hook contents
        hooks_file = workspace / ".claude" / "hooks.json"
        self.assertTrue(hooks_file.is_file())
        hooks_installed = json.loads(hooks_file.read_text(encoding="utf-8"))

        # Verify command hook with args
        self.assertEqual(
            hooks_installed["cmd-guard"]["PreToolUse"][0]["matcher"], "Read")
        self.assertEqual(
            hooks_installed["cmd-guard"]["PreToolUse"][0]["hooks"][0]
            ["command"],
            compiler.fail_safe_hook_command(
                "python3 ~/.tpu-agent-suite/packages/pkg-hooks@1.0.0/framework/runtime.py hook-dispatch "
                "--harness claude-code --package pkg-hooks@1.0.0 --hook cmd-guard "
                "--event PreToolUse --timeout 5 --on-failure deny --command 'python3 check_input.py --strict'",
                "cmd-guard", "deny", "PreToolUse"),
        )
        self.assertEqual(
            hooks_installed["cmd-guard"]["PreToolUse"][0]["hooks"][0]
            ["timeout"], 5 + compiler.HOOK_DISPATCH_HEADROOM_S)

        # Verify custom script hook & script file presence
        self.assertEqual(
            hooks_installed["custom-guard"]["PreToolUse"][0]["hooks"][0]
            ["command"],
            compiler.fail_safe_hook_command(
                "python3 ~/.tpu-agent-suite/packages/pkg-hooks@1.0.0/framework/runtime.py hook-dispatch "
                "--harness claude-code --package pkg-hooks@1.0.0 --hook custom-guard "
                "--event PreToolUse --timeout 10 --on-failure deny --script scripts/custom_guard.py",
                "custom-guard", "deny", "PreToolUse"),
        )
        central_script = (self.central_home / "packages" / "pkg-hooks@1.0.0" /
                          "scripts" / "custom_guard.py")
        self.assertTrue(central_script.is_file())

        # Verify ownership tracking
        ownership = json.loads((workspace / ".tpu-agent-suite" /
                                "ownership.json").read_text(encoding="utf-8"))
        tracked_dests = [
            entry["path"] for entry in ownership.get("actions", [])
        ]
        self.assertIn(".claude/hooks.json", tracked_dests)
        # Central payload is governed by the reference ledger, not per-file
        # ownership actions.
        self.assertNotIn(
            ".tpu-agent-suite/packages/pkg-hooks@1.0.0/scripts/custom_guard.py",
            tracked_dests)
        self.assertEqual(ownership.get("central_package"), "pkg-hooks@1.0.0")

        # 2. Test user modification preservation on uninstall
        hooks_file.write_text(json.dumps({"custom": "user-modified"}),
                              encoding="utf-8")
        modified = installer.uninstall(package="pkg-hooks",
                                       destination=workspace)
        self.assertIn(".claude/hooks.json", modified)
        self.assertTrue(
            hooks_file.is_file())  # Preserved because user modified it
        self.assertFalse(central_script.exists())  # GC'd: last reference gone

    def test_upgrade_replaces_previous_versions_hook_entries(self):
        """A version bump changes every hook command (it names the version).

        Those entries were added by the installer, so an upgrade must treat
        them as its own to replace, not as the user's settings to refuse.
        Regression: 0.3.0 -> 0.4.0 of tpu-agent-suite failed with
        'refusing to replace existing JSON setting: hooks/PostToolUse'.
        """
        pkg_dir = self._create_pkg_with_hooks()
        manifest_path = pkg_dir / "package.release.json"
        dists = {}
        for version in ("1.0.0", "2.0.0"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["version"] = version
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            dists[version] = self.temp_dir / f"dist_hooks_{version}"
            compiler.translate(pkg_dir, "claude-code", dists[version])

        workspace = self.temp_dir / "workspace_hook_upgrade"
        settings = workspace / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True)
        # A setting of the user's own, beside the package's.
        settings.write_text(json.dumps(
            {"permissions": {
                "allow": ["Bash(ls:*)"]
            }}),
                            encoding="utf-8")
        install = functools.partial(installer.install,
                                    destination=workspace,
                                    non_interactive=True,
                                    skip_harness_check=True)
        install(distribution=dists["1.0.0"])
        install(distribution=dists["2.0.0"])

        document = json.loads(settings.read_text(encoding="utf-8"))
        commands = [
            hook["command"] for entry in document["hooks"]["PreToolUse"]
            for hook in entry["hooks"]
        ]
        self.assertTrue(commands)
        self.assertTrue(all("pkg-hooks@2.0.0" in c for c in commands))
        self.assertFalse(any("pkg-hooks@1.0.0" in c for c in commands))
        self.assertEqual(document["permissions"], {"allow": ["Bash(ls:*)"]})

    def test_rollback_json_records_undoes_only_what_the_ledger_added(self):
        document = {
            "hooks": {
                "PreToolUse": ["pkg@1"],
                "Notification": ["mine"],
            },
            "env": {
                "PKG_VAR": "1",
                "MY_VAR": "edited"
            },
        }
        ownership = {
            "actions": [
                {
                    "kind": "json_container",
                    "keys": ["hooks"],
                    "path": ".claude/settings.json"
                },
                {
                    "kind": "json_value",
                    "keys": ["hooks", "PreToolUse"],
                    "value": ["pkg@1"],
                    "path": ".claude/settings.json"
                },
                {
                    "kind": "json_container",
                    "keys": ["env"],
                    "path": ".claude/settings.json"
                },
                {
                    "kind": "json_value",
                    "keys": ["env", "PKG_VAR"],
                    "value": "1",
                    "path": ".claude/settings.json"
                },
                {
                    "kind": "json_value",
                    "keys": ["env", "MY_VAR"],
                    "value": "original",
                    "path": ".claude/settings.json"
                },
                {
                    "kind": "json_value",
                    "keys": ["hooks", "Notification"],
                    "value": ["mine"],
                    "path": "other.json"
                },
            ]
        }
        installer.rollback_json_records(document, ownership,
                                        ".claude/settings.json")
        # The user's Notification hook (not in this file's ledger) and the
        # edited MY_VAR stay; the package's entries go; `hooks` and `env`
        # stay because they are not empty.
        self.assertEqual(document, {
            "hooks": {
                "Notification": ["mine"]
            },
            "env": {
                "MY_VAR": "edited"
            },
        })

    def test_install_and_uninstall_antigravity_hooks(self):
        pkg_dir = self._create_pkg_with_hooks()
        dist_dir = self.temp_dir / "dist_ag"
        compiler.translate(pkg_dir, "antigravity", dist_dir)

        workspace = self.temp_dir / "workspace_ag"
        workspace.mkdir()

        installer.install(dist_dir,
                          destination=workspace,
                          non_interactive=True)

        # Verify hooks.json and plugin.json installed as real files
        hooks_json = workspace / ".agents" / "hooks.json"
        self.assertTrue(hooks_json.is_file())
        plugin_json = workspace / ".agents" / "plugin.json"
        self.assertTrue(plugin_json.is_file())
        plugin_manifest = json.loads(plugin_json.read_text(encoding="utf-8"))
        self.assertEqual(plugin_manifest["name"], "pkg-hooks")

        # Verify zero symlinks created
        for p in workspace.rglob("*"):
            self.assertFalse(p.is_symlink(), f"Unexpected symlink found: {p}")

        # Verify ownership tracks only files
        ownership = json.loads((workspace / ".tpu-agent-suite" /
                                "ownership.json").read_text(encoding="utf-8"))
        self.assertTrue(
            all(a["kind"] == "file" for a in ownership.get("actions", [])))

        # Uninstall and verify clean removal
        installer.uninstall(package="pkg-hooks", destination=workspace)
        self.assertFalse(hooks_json.exists())
        self.assertFalse(plugin_json.exists())
        self.assertFalse((workspace / ".tpu-agent-suite" / "packages" /
                          "pkg-hooks@1.0.0").exists())

    def test_install_and_uninstall_codex_hooks(self):
        pkg_dir = self._create_pkg_with_hooks()
        dist_dir = self.temp_dir / "dist_codex_hooks"
        compiler.translate(pkg_dir, "codex", dist_dir)

        workspace = self.temp_dir / "workspace_codex"
        workspace.mkdir()

        installer.install(dist_dir,
                          destination=workspace,
                          non_interactive=True,
                          skip_harness_check=True)

        # Verify hooks.json installed in .codex
        hooks_json = workspace / ".codex" / "hooks.json"
        self.assertTrue(hooks_json.is_file())

        # Verify config.toml has [hooks] enabled = true
        config_toml = (workspace / ".codex" /
                       "config.toml").read_text(encoding="utf-8")
        self.assertIn("[hooks]", config_toml)
        self.assertIn("enabled = true", config_toml)

        # Verify ownership tracks hooks.json
        ownership = json.loads((workspace / ".tpu-agent-suite" /
                                "ownership.json").read_text(encoding="utf-8"))
        tracked_dests = [
            entry["path"] for entry in ownership.get("actions", [])
        ]
        self.assertIn(".codex/hooks.json", tracked_dests)

        # Uninstall and verify clean removal
        installer.uninstall(package="pkg-hooks", destination=workspace)
        self.assertFalse(hooks_json.exists())
        self.assertFalse((workspace / ".tpu-agent-suite" / "packages" /
                          "pkg-hooks@1.0.0").exists())

    def test_harness_install_destinations_all_scopes(self):
        pkg_dir = self._create_pkg_with_hooks()

        expected = {
            "claude-code": {
                "project": ".claude",
                "user": ".claude",
            },
            "codex": {
                "project": ".codex",
                "user": ".codex",
            },
            "antigravity": {
                "project": ".agents",
                "user": ".gemini/config",
            },
        }

        for harness, scopes in expected.items():
            for scope, expected_dir in scopes.items():
                with self.subTest(harness=harness, scope=scope):
                    dist_dir = self.temp_dir / f"dist_{harness}_{scope}"
                    compiler.translate(pkg_dir, harness, dist_dir)

                    target_dir = self.temp_dir / f"target_{harness}_{scope}"
                    target_dir.mkdir()

                    with mock.patch("framework.installer.Path.cwd", return_value=target_dir), \
                         mock.patch("framework.installer.Path.home", return_value=target_dir):

                        dest = target_dir if scope == "project" else None
                        installer.install(dist_dir,
                                          destination=dest,
                                          scope=scope,
                                          non_interactive=True,
                                          skip_harness_check=True)

                        hooks_json = target_dir / expected_dir / "hooks.json"
                        self.assertTrue(hooks_json.is_file(),
                                        f"Expected {hooks_json} to be a file")


class TestHostCheck(unittest.TestCase):
    """The software check install runs on this machine."""

    def setUp(self):
        super().setUp()
        self.dist = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dist)
        (self.dist / installer.ENV_CHECK).parent.mkdir(parents=True)
        for name in (installer.ENV_CHECK, installer.HOST_REQUIREMENTS):
            (self.dist / name).write_text("", encoding="utf-8")

    def test_the_host_check_runs_under_python3_and_never_blocks(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("shutil.which", return_value="/usr/bin/python3"), \
                mock.patch("subprocess.run",
                           return_value=subprocess.CompletedProcess(
                               [], 1, "[FAIL] pyyaml", "")) as run, \
                contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            installer.check_host(self.dist)
        self.assertEqual(run.call_args.args[0], [
            "/usr/bin/python3",
            str(self.dist / installer.ENV_CHECK), "--host", "--requirements",
            str(self.dist / "requirements-client.txt")
        ])
        self.assertIn("[FAIL] pyyaml", out.getvalue())
        self.assertIn("WARNING:", err.getvalue())
        self.assertIn("the install goes on", err.getvalue())


class TestWorkspaceMemory(unittest.TestCase):
    """Tests for managed blocks installed into the workspace memory file."""

    MARKER = "tpu-agent-suite:pkg:kb"

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        self.central_home.mkdir()
        self.old_tpu_agent_suite_home = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = str(self.central_home)
        self.pkg_dir = TestInstaller._create_test_package(self)

        manifest = json.loads(
            (self.pkg_dir /
             "package.release.json").read_text(encoding="utf-8"))
        manifest["workspace_memory"] = [{
            "marker": "kb",
            "content": "memory/contract.md",
        }]
        manifest["compatibility_targets"] += [
            {
                "name": "antigravity",
                "harness": "antigravity",
                "versions": ">=1.0.0",
            },
            {
                "name": "codex",
                "harness": "codex",
                "versions": ">=0.145.0",
            },
        ]
        (self.pkg_dir / "package.release.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        (self.pkg_dir / "memory").mkdir(parents=True, exist_ok=True)
        (self.pkg_dir / "memory" / "contract.md").write_text(
            "Schemas live at {{schema:in.json}}.", encoding="utf-8")

        self.dist_dir = self.temp_dir / "dist_memory"
        compiler.translate(self.pkg_dir, "claude-code", self.dist_dir)

    def tearDown(self):
        if self.old_tpu_agent_suite_home is not None:
            os.environ["TPU_AGENT_SUITE_HOME"] = self.old_tpu_agent_suite_home
        else:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _install(self, workspace: Path, **kwargs) -> None:
        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
            **kwargs,
        )

    def _workspace(self, name: str) -> Path:
        workspace = self.temp_dir / name
        workspace.mkdir(exist_ok=True)
        return workspace

    def test_block_created_with_resolved_path(self):
        """An absent memory file is created holding the resolved block."""
        workspace = self._workspace("ws_mem_new")
        self._install(workspace)

        memory = workspace / "CLAUDE.md"
        self.assertTrue(memory.is_file())
        content = memory.read_text(encoding="utf-8")
        self.assertIn(f"<!-- {self.MARKER}:begin -->", content)
        self.assertIn(f"<!-- {self.MARKER}:end -->", content)
        # Central-store assets render at compile time as ~-anchored paths,
        # which hold from any cwd and any machine.
        pkg_tag = utils.format_package_tag("pkg", "1.0.0")
        self.assertIn(f"~/.tpu-agent-suite/packages/{pkg_tag}/schemas/",
                      content)
        self.assertNotIn("{install_root}", content)

    def test_user_scope_writes_where_claude_code_reads_user_memory(self):
        """~/.claude/CLAUDE.md, not ~/CLAUDE.md, which only a session started
        in $HOME itself would read."""
        home = self._workspace("home")
        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            installer.install(
                distribution=self.dist_dir,
                scope="user",
                non_interactive=True,
                skip_harness_check=True,
            )
        memory = home / ".claude" / "CLAUDE.md"
        self.assertIn(f"<!-- {self.MARKER}:begin -->",
                      memory.read_text(encoding="utf-8"))
        self.assertFalse((home / "CLAUDE.md").exists())

    def _install_user_scope(self, harness: str) -> Path:
        """Compiles for `harness`, installs at user scope; returns $HOME."""
        dist_dir = self.temp_dir / f"dist_memory_{harness}"
        compiler.translate(self.pkg_dir, harness, dist_dir)
        home = self._workspace(f"home_{harness}")
        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            installer.install(
                distribution=dist_dir,
                scope="user",
                non_interactive=True,
                skip_harness_check=True,
            )
        return home

    def test_user_scope_writes_where_codex_reads_user_memory(self):
        """~/.codex/AGENTS.md, not ~/AGENTS.md, which only a session started
        in $HOME itself would read."""
        home = self._install_user_scope("codex")
        memory = home / ".codex" / "AGENTS.md"
        self.assertIn(f"<!-- {self.MARKER}:begin -->",
                      memory.read_text(encoding="utf-8"))
        self.assertFalse((home / "AGENTS.md").exists())

    def test_user_scope_writes_where_antigravity_reads_user_memory(self):
        """~/.gemini/AGENTS.md, not ~/AGENTS.md, which only a session started
        in $HOME itself would read."""
        home = self._install_user_scope("antigravity")
        memory = home / ".gemini" / "AGENTS.md"
        self.assertIn(f"<!-- {self.MARKER}:begin -->",
                      memory.read_text(encoding="utf-8"))
        self.assertFalse((home / "AGENTS.md").exists())

    def test_relative_path_for_project_scope(self):
        """Project scope resolves the block path relative to the workspace."""
        workspace = self._workspace("ws_mem_scope")
        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            scope="project",
            non_interactive=True,
            skip_harness_check=True,
        )

        content = (workspace / "CLAUDE.md").read_text(encoding="utf-8")
        pkg_tag = utils.format_package_tag("pkg", "1.0.0")
        # Every scope gets the same compile-time ~-anchored central path.
        self.assertIn(f"~/.tpu-agent-suite/packages/{pkg_tag}/schemas/",
                      content)
        self.assertNotIn(str(workspace), content)

    def test_reinstall_is_idempotent(self):
        """Re-installing replaces the block instead of appending a second one."""
        workspace = self._workspace("ws_mem_idem")
        self._install(workspace)
        first = (workspace / "CLAUDE.md").read_text(encoding="utf-8")
        self._install(workspace, upgrade=True)
        second = (workspace / "CLAUDE.md").read_text(encoding="utf-8")

        self.assertEqual(first, second)
        self.assertEqual(second.count(f"<!-- {self.MARKER}:begin -->"), 1)

    def test_uninstall_preserves_user_content(self):
        """Uninstall strips only the managed block from a user-owned file."""
        workspace = self._workspace("ws_mem_user")
        memory = workspace / "CLAUDE.md"
        memory.write_text("# Project\n\nMy notes.\n", encoding="utf-8")

        self._install(workspace)
        self.assertIn("My notes.", memory.read_text(encoding="utf-8"))

        installer.uninstall(destination=workspace, package="pkg")
        remaining = memory.read_text(encoding="utf-8")
        self.assertIn("My notes.", remaining)
        self.assertNotIn(self.MARKER, remaining)

    def test_uninstall_removes_created_file(self):
        """A memory file created by the install is deleted when nothing remains."""
        workspace = self._workspace("ws_mem_created")
        self._install(workspace)
        self.assertTrue((workspace / "CLAUDE.md").is_file())

        installer.uninstall(destination=workspace, package="pkg")
        self.assertFalse((workspace / "CLAUDE.md").exists())

    def test_refuses_unmanaged_existing_block(self):
        """An unowned block with the same marker blocks the install."""
        workspace = self._workspace("ws_mem_unmanaged")
        (workspace / "CLAUDE.md").write_text(
            f"<!-- {self.MARKER}:begin -->\nstale\n<!-- {self.MARKER}:end -->\n",
            encoding="utf-8",
        )

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            self._install(workspace)
        self.assertIn("unmanaged block", str(ctx.exception))

    def test_detects_local_block_edit(self):
        """A hand-edited block is reported as modified managed content."""
        workspace = self._workspace("ws_mem_edited")
        self._install(workspace)
        memory = workspace / "CLAUDE.md"
        memory.write_text(
            memory.read_text(encoding="utf-8").replace("Schemas live",
                                                       "Edited"),
            encoding="utf-8",
        )

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            self._install(workspace, upgrade=True)
        self.assertIn("managed content changed", str(ctx.exception))

    def test_force_overwrites_edited_block(self):
        """--force replaces a locally edited block."""
        workspace = self._workspace("ws_mem_force")
        self._install(workspace)
        memory = workspace / "CLAUDE.md"
        memory.write_text(
            memory.read_text(encoding="utf-8").replace("Schemas live",
                                                       "Edited"),
            encoding="utf-8",
        )

        self._install(workspace, upgrade=True, force=True)
        self.assertIn("Schemas live", memory.read_text(encoding="utf-8"))

    def test_workspace_memory_can_be_skipped(self):
        """--no-workspace-memory installs the package without touching memory."""
        workspace = self._workspace("ws_mem_skip")
        self._install(workspace, workspace_memory=False)

        self.assertFalse((workspace / "CLAUDE.md").exists())
        ownership = utils.load_json(workspace / ".tpu-agent-suite" /
                                    "ownership.json")
        kinds = {action["kind"] for action in ownership["actions"]}
        self.assertNotIn("markdown_block", kinds)

    def test_resolve_config_merge_path_across_harnesses_and_scopes(self):
        """Verify resolution of settings files across Claude, Cursor, and Antigravity."""
        # Claude Code
        self.assertEqual(
            installer.resolve_config_merge_path("claude-code",
                                                ".claude/settings.local.json",
                                                "project"),
            ".claude/settings.json",
        )
        self.assertEqual(
            installer.resolve_config_merge_path("claude-code",
                                                ".claude/settings.json",
                                                "project"),
            ".claude/settings.json",
        )
        self.assertEqual(
            installer.resolve_config_merge_path("claude-code",
                                                ".claude/settings.local.json",
                                                "user"),
            ".claude/settings.json",
        )
        self.assertEqual(
            installer.resolve_config_merge_path("claude-code",
                                                ".claude/settings.json",
                                                "user"),
            ".claude/settings.json",
        )

        # Cursor
        self.assertEqual(
            installer.resolve_config_merge_path("cursor",
                                                ".cursor/settings.json",
                                                "project"),
            ".cursor/settings.json",
        )
        self.assertEqual(
            installer.resolve_config_merge_path("cursor",
                                                ".cursor/settings.json",
                                                "user"),
            ".cursor/settings.json",
        )

        # Antigravity
        self.assertEqual(
            installer.resolve_config_merge_path("antigravity",
                                                ".agents/settings.json",
                                                "project"),
            ".agents/settings.json",
        )
        self.assertEqual(
            installer.resolve_config_merge_path("antigravity",
                                                ".agents/settings.json",
                                                "user"),
            ".agents/settings.json",
        )

        # Unrelated paths
        self.assertEqual(
            installer.resolve_config_merge_path("claude-code",
                                                ".claude/mcp.json", "project"),
            ".claude/mcp.json",
        )

    def test_config_merge_installation_and_cleanup(self):
        """Test that installing with project scope merges into settings.json and records ownership."""
        workspace = self._workspace("ws_scope_project")
        installer.install(
            distribution=self.dist_dir,
            destination=workspace,
            scope="project",
            non_interactive=True,
            skip_harness_check=True,
        )
        # In project scope, writes .claude/settings.json
        project_settings = workspace / ".claude" / "settings.json"
        self.assertTrue(project_settings.exists())
        self.assertFalse(
            (workspace / ".claude" / "settings.local.json").exists())

        # Ownership records settings.json
        ownership = utils.load_json(workspace / ".tpu-agent-suite" /
                                    "ownership.json")
        paths = {action.get("path") for action in ownership["actions"]}
        self.assertIn(".claude/settings.json", paths)
        self.assertEqual(ownership.get("scope"), "project")


if __name__ == "__main__":
    unittest.main()
