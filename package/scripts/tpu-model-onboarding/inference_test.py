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
import os
import unittest

import fastapi.testclient

# pylint: disable=g-importing-member

# Create a temporary dummy configuration to satisfy module-level config loading
CONFIG_PATH = "deployments.json"
DUMMY_CONFIG = {
    "deployments": [{
        "model_id": "google/gemma-4-12B",
        "engine": "custom_fastapi",
        "port": 8000,
        "max_model_len": 2048,
    }]
}


def setup_dummy_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DUMMY_CONFIG, f)


setup_dummy_config()

# Importing after dummy config exists so module level load_config()
# executes safely
try:
    import inference  # pylint: disable=g-import-not-at-top
except ImportError as e:
    raise RuntimeError(f"Error importing inference.py: {e}") from e


class TestInferenceServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Create a FastAPI test client bridging the app
        cls.client = fastapi.testclient.TestClient(inference.app)

    def test_config_parsing(self):
        """Verifies Pydantic appropriately validates deployments.json."""
        config = inference.load_config()
        self.assertIsInstance(config, inference.FastAPIDeployment)
        self.assertEqual(config.model_id, "google/gemma-4-12B")
        self.assertEqual(config.port, 8000)

    def test_device_fallback(self):
        """Verifies the device identifier utility operates cleanly."""
        cpu_device = inference.get_device("cpu")
        self.assertEqual(cpu_device.type, "cpu")

    def test_health_endpoint(self):
        """Verifies the FastAPI web server initializes and endpoints are open."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_pipeline_endpoint(self):
        """>>> AGENT TODO: Implement a test for your pipeline-specific endpoint.

    (e.g., /generate, /embed).
    Send a test payload using self.client and verify it returns a HTTP
    200 OK.
    Note: You may need to mock the Hugging Face model/tokenizer
    initialization and forward pass to avoid loading actual
    multi-gigabyte model weights during this unit test execution.
    """
        # Example for Text Generation:
        # response = self.client.post(
        #     "/generate", json={"prompt": "Hello", "max_tokens": 10}
        # )
        # self.assertEqual(response.status_code, 200)
        # self.assertIn("result", response.json())


if __name__ == "__main__":
    unittest.main()
