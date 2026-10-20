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
"""Which configs are control arms, and that none still selects skills."""

import unittest
from pathlib import Path

import yaml

from eval.utils import (find_repo_root, harness_for_config, is_control_arm,
                        released_skills, skill_subset)

CONFIGS = find_repo_root() / "eval" / "configs"

# Written out, so a control arm named without `baseline` -- which would
# quietly install the whole package -- fails here rather than in a score.
CONTROL_ARMS = {
    "agy/baseline.yaml",
    "agy/baseline-remote.yaml",
    "agy/baseline-shared-capacity-remote.yaml",
    "claude-code/baseline.yaml",
    "claude-code/baseline-remote.yaml",
    "claude-code/baseline-shared-capacity-remote.yaml",
}


class TestIsControlArm(unittest.TestCase):

    def configs(self) -> list[Path]:
        return sorted(CONFIGS.rglob("*.yaml"))

    def test_exactly_the_baselines_are_control_arms(self) -> None:
        found = {
            str(p.relative_to(CONFIGS))
            for p in self.configs() if is_control_arm(p)
        }
        self.assertEqual(CONTROL_ARMS, found)

    def test_a_baseline_part_is_needed_not_a_substring(self) -> None:
        self.assertTrue(is_control_arm("x/true_baseline_gke.yaml"))
        self.assertTrue(is_control_arm("x/baseline-opus.yaml"))
        self.assertTrue(is_control_arm("x/opus-Baseline.yaml"))
        self.assertFalse(is_control_arm("x/baselinez_v1.yaml"))
        self.assertFalse(is_control_arm("baseline/agent-suite.yaml"))

    def test_each_pair_matches_apart_from_job_name(self) -> None:
        for harness in ("claude-code", "agy"):
            configs = {
                name:
                yaml.safe_load(
                    (CONFIGS / harness / f"{name}.yaml").read_text())
                for name in ("agent-suite", "baseline", "agent-suite-remote",
                             "baseline-remote",
                             "agent-suite-shared-capacity-remote",
                             "baseline-shared-capacity-remote")
            }
            for config in configs.values():
                config.pop("job_name")
            for base, name in (("agent-suite", "baseline"),
                               ("agent-suite-remote", "baseline-remote"),
                               ("agent-suite-shared-capacity-remote",
                                "baseline-shared-capacity-remote")):
                with self.subTest(harness=harness, config=name):
                    self.assertEqual(configs[base], configs[name])

    def test_each_remote_config_matches_its_local_one_apart_from_where(
            self) -> None:
        for path in self.configs():
            if not path.stem.endswith("-remote"):
                continue
            local_path = path.with_name(
                path.stem.removesuffix("-remote") + ".yaml")
            if not local_path.exists():
                local_path = path.with_name(
                    path.stem.replace("baseline-", "agent-suite-",
                                      1).removesuffix("-remote") + ".yaml")
            remote = yaml.safe_load(path.read_text())
            local = yaml.safe_load(local_path.read_text())
            with self.subTest(config=str(path.relative_to(CONFIGS))):
                self.assertEqual("gke", remote["environment"].pop("type"))
                for key in ("force_build", "kwargs"):
                    remote["environment"].pop(key)
                for key in ("agent_setup_timeout_multiplier",
                            "timeout_multiplier"):
                    remote.pop(key)
                for config in (remote, local):
                    config.pop("job_name")
                self.assertEqual(local, remote)

    def test_every_agy_config_authenticates_with_adc(self) -> None:
        adc = "/root/.config/gcloud/application_default_credentials.json"
        for path in sorted((CONFIGS / "agy").glob("*.yaml")):
            config = yaml.safe_load(path.read_text())
            environment = config["environment"]
            with self.subTest(config=path.name):
                self.assertIn(adc,
                              [m["target"] for m in environment["mounts"]])
                self.assertEqual(
                    adc, config["agents"][0]["env"]
                    ["GOOGLE_APPLICATION_CREDENTIALS"])
                self.assertEqual("true", environment["env"]["AGY_ADC_AUTH"])

    def test_no_config_selects_skills(self) -> None:
        for path in self.configs():
            config = yaml.safe_load(path.read_text()) or {}
            with self.subTest(config=str(path.relative_to(CONFIGS))):
                for key in ("stage_skills", "packages", "package_dir"):
                    self.assertNotIn(key, config)
                for agent in config.get("agents") or []:
                    self.assertNotIn("package_dir", agent.get("kwargs") or {})


class TestHarnessForConfig(unittest.TestCase):

    def test_each_config_directory_is_its_harness(self) -> None:
        for directory, harness in (("claude-code", "claude-code"),
                                   ("agy", "antigravity")):
            for path in sorted((CONFIGS / directory).glob("*.yaml")):
                with self.subTest(config=str(path.relative_to(CONFIGS))):
                    self.assertEqual(
                        harness,
                        harness_for_config(yaml.safe_load(path.read_text())))


class TestSkillSubset(unittest.TestCase):
    """The subset mode's skills are checked against what package.release.json releases."""

    RELEASED = ["tpu-agent", "tpu-query-understanding"]

    def test_comma_and_space_separated_are_the_same(self) -> None:
        self.assertEqual(
            self.RELEASED,
            skill_subset(["tpu-query-understanding,tpu-agent"], self.RELEASED))
        self.assertEqual(
            self.RELEASED,
            skill_subset(["tpu-agent", "tpu-query-understanding"],
                         self.RELEASED))

    def test_an_unknown_or_empty_subset_is_refused(self) -> None:
        for names in (["tpu-nope"], [","]):
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    skill_subset(names, self.RELEASED)

    def test_the_released_skills_include_the_entrypoint(self) -> None:
        self.assertIn("tpu-agent", released_skills())


if __name__ == "__main__":
    unittest.main()
