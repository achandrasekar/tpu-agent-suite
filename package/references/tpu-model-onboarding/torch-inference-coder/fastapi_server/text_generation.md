---
scripts:
  - scripts/tpu-model-onboarding/inference.py
---
# Recipe: Text Generation (Causal LM)

When deploying a Causal LM (e.g. LLaMA, Mistral) on `custom_fastapi`, inherit
the `{{script:tpu-model-onboarding/inference.py}}` template and implement the logic blocks below.

### 1. Model Initialization
Replace the `load_model()` stub, and import beside it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    model = AutoModelForCausalLM.from_pretrained(
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
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    ignore_eos: bool = False
```

`verify_inference.py` sends `temperature: 0` so its verdict is the same on
every run; keep the field so that request is honoured. The serving benchmark
sends `ignore_eos: true` so every request generates exactly `max_tokens`, as
`vllm bench serve --ignore-eos` does; keep that field too.

### 3. Generate Endpoint API
Replace the `/generate` endpoint block:

```python
@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        prompt = req.prompt
        if getattr(tokenizer, "chat_template", None):
            # Instruct models answer the template they were tuned on; a raw
            # string can make them ramble or loop.
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": req.prompt}],
                tokenize=False,
                add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        sampling = ({"do_sample": True, "temperature": req.temperature}
                    if req.temperature > 0 else {"do_sample": False})
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                # Generating past end-of-sequence holds every benchmark
                # request to the same length.
                min_new_tokens=req.max_tokens if req.ignore_eos else None,
                **sampling
            )
        # Decode only the new tokens: the prompt is not the model's answer.
        prompt_len = inputs["input_ids"].shape[-1]
        generated_text = tokenizer.decode(outputs[0][prompt_len:],
                                          skip_special_tokens=True)
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
    @patch('inference.AutoModelForCausalLM.from_pretrained')
    @patch('inference.AutoTokenizer.from_pretrained')
    def test_pipeline_endpoint(self, mock_tokenizer, mock_model, _mock_fetch):
        """Verifies the text generation endpoint parses payloads gracefully."""
        # Setup mock behavior
        mock_model.return_value.to.return_value.generate.return_value = [[1, 2, 3]]
        tok = mock_tokenizer.return_value
        tok.chat_template = None
        tok.return_value.to.return_value = {"input_ids": MagicMock(shape=(1, 2))}
        tok.decode.return_value = "Mocked generated text"
        
        # Dispatch test payload
        # Ensure your payload aligns strictly with GenerateRequest schema
        # `with` runs the lifespan, so load_model() loads the mocks.
        with fastapi.testclient.TestClient(inference.app) as client:
            response = client.post("/generate", json={"prompt": "Hello", "max_tokens": 10})
        self.assertEqual(response.status_code, 200)
        self.assertIn("result", response.json())
        self.assertEqual(response.json()["result"], "Mocked generated text")
```
