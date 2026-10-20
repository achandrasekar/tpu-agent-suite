---
scripts:
  - scripts/tpu-model-onboarding/inference.py
---
# Recipe: Vision-Language Models (VLM)

When deploying a multi-modal endpoint on `custom_fastapi`, inherit the
`{{script:tpu-model-onboarding/inference.py}}` template and implement the logic blocks below.

### 1. Model Initialization
Replace the `load_model()` stub, and import beside it:

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

from fetch_weights import fetch

device = get_device('tpu')
processor = model = None


def load_model() -> None:
    global processor, model
    # Downloads what the model loads and prints WEIGHTS PROGRESS / WEIGHTS
    # READY; Task 3b waits on that line. Load from the local path it returns.
    weights = fetch(config.model_id)
    print(f"Loading {config.model_id} from {weights} on {device}...")

    # local_files_only: everything is in `weights`; a missing file is an
    # error, never a download.
    processor = AutoProcessor.from_pretrained(
        weights, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        weights,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
```

Keep the loading inside `load_model()`. The server runs it at startup;
`inference_test.py` imports the module without it, so the test downloads
nothing.

### 2. Payload Schema
Replace the `InferenceRequest` block:

```python
class VLMRequest(BaseModel):
    prompt: str
    image_url: str
    max_tokens: int = 128
    ignore_eos: bool = False
```

The serving benchmark sends `ignore_eos: true` so every request generates
exactly `max_tokens`; keep the field.

### 3. Generate Endpoint API
Replace the `/generate` endpoint block with a vision-text payload logic:

```python
import requests
from io import BytesIO
from PIL import Image

@app.post("/vqa")
def vqa(req: VLMRequest):
    try:
        response = requests.get(req.image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        inputs = processor(text=req.prompt, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                min_new_tokens=req.max_tokens if req.ignore_eos else None)

        # Decode only the new tokens: the prompt is not the model's answer,
        # and left in, the benchmark would count it as output.
        prompt_len = inputs["input_ids"].shape[-1]
        generated_text = processor.batch_decode(
            [output[prompt_len:] for output in outputs],
            skip_special_tokens=True)[0]
        return {"result": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 4. End-to-End API Test
Modify the `inference_test.py` file to replace the `AGENT TODO` block inside
`test_pipeline_endpoint` with a mock generation test (add
`from unittest.mock import MagicMock, patch` to its imports):

```python
    @patch('inference.fetch', return_value='/mock/weights')
    @patch('inference.AutoModelForVision2Seq.from_pretrained')
    @patch('inference.AutoProcessor.from_pretrained')
    def test_pipeline_endpoint(self, mock_processor, mock_model, _mock_fetch):
        """Verifies the text generation endpoint parses payloads gracefully."""
        # Setup mock behavior
        mock_model.return_value.to.return_value.generate.return_value = [[1, 2, 3]]
        proc = mock_processor.return_value
        proc.return_value.to.return_value = {"input_ids": MagicMock(shape=(1, 2))}
        proc.batch_decode.return_value = ["Mocked VLM text"]

        # Dispatch test payload
        # Ensure your payload aligns strictly with VLMRequest schema
        # `with` runs the lifespan, so load_model() loads the mocks.
        with fastapi.testclient.TestClient(inference.app) as client:
            response = client.post("/vqa", json={
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/e/e1/FullMoon2010.jpg",
                "prompt": "What is in this image?"
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn("result", response.json())
        self.assertEqual(response.json()["result"], "Mocked VLM text")
        # Only the token after the 2-token prompt is decoded.
        proc.batch_decode.assert_called_once_with([[3]], skip_special_tokens=True)
```
