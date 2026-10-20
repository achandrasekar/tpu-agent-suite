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

import json
import pathlib

import setuptools

_pkg = pathlib.Path(
    __file__).resolve().parents[1] / "package" / "package.release.json"
_version = json.loads(_pkg.read_text(
    encoding="utf-8")).get("version", "0.0.0") if _pkg.is_file() else "0.0.0"

setuptools.setup(version=_version)
