---
scripts:
  - scripts/tpu-model-onboarding/inference.py
---
# Recipe: Embeddings

When deploying a feature extraction encoder on `custom_fastapi`, inherit the
`{{script:tpu-model-onboarding/inference.py}}` template and implement the logic blocks below.

### 1. Model Initialization
Replace the `load_model()` stub, and import beside it:

```python
from transformers import AutoModel, AutoTokenizer
import torch

from fetch_weights import fetch

device = get_device('tpu')
tokenizer = model = None


def load_model() -> None:
    global tokenizer, model
    # Downloads what the model loads and prints WEIGHTS PROGRESS / WEIGHTS
    # READY; Task 3b waits on that line. Load from the local path it returns.
    weights = fetch(config.model_id)
    print(f"Loading {config.model_id} from {weights} on {device}...")

    # local_files_only: everything is in `weights`; a missing file is an
    # error, never a download.
    tokenizer = AutoTokenizer.from_pretrained(
        weights, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
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
class EmbeddingRequest(BaseModel):
    text: str
```

### 3. Generate Endpoint API
Replace the `/generate` endpoint block with an `/embed` payload endpoint:

```python
@app.post("/embed")
def embed(req: EmbeddingRequest):
    try:
        inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Example pooling output
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return {"embedding": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 4. End-to-End API Test
Modify the `inference_test.py` file to replace the `AGENT TODO` block inside
`test_pipeline_endpoint` with a mock generation test:

```python
    @patch('inference.fetch', return_value='/mock/weights')
    @patch('inference.AutoModel.from_pretrained')
    @patch('inference.AutoTokenizer.from_pretrained')
    def test_pipeline_endpoint(self, mock_tokenizer, mock_model, _mock_fetch):
        """Verifies the embedding endpoint parses payloads gracefully."""
        # Setup mock behavior
        import torch
        mock_model.return_value.to.return_value = mock_model.return_value

        # We need mock_model(**inputs) to return an object with last_hidden_state
        class MockOutput:
            last_hidden_state = torch.rand(1, 4, 768)

        mock_model.return_value.side_effect = lambda **kwargs: MockOutput()

        # Dispatch test payload
        # Ensure your payload aligns strictly with EmbedRequest schema
        # `with` runs the lifespan, so load_model() loads the mocks.
        with fastapi.testclient.TestClient(inference.app) as client:
            response = client.post("/embed", json={"text": "Hello world"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("embedding", response.json())
        self.assertIsInstance(response.json()["embedding"], list)
```
