# Wav2Vec2 / Audio Speech Recipe
## Routing Rules
Apply this recipe if:

*   `pipeline_tag` is `automatic-speech-recognition`
*   **AND** `model_id` does **NOT** contain "whisper" (case-insensitive)

For audio models that process raw waveform inputs (e.g., Wav2Vec2, HuBERT,
WavLM):

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """Wav2Vec2 benchmark forward logic."""
  transformers = import_dependency("transformers")
  np = import_dependency("numpy")

  AutoModel = getattr(transformers, "AutoModel")
  AutoFeatureExtractor = getattr(transformers, "AutoFeatureExtractor")

  print(f"Loading {model_id}...")
  model, feature_extractor = load_model_and_processor(
      model_id, device, AutoModel, AutoFeatureExtractor, use_bfloat16=True
  )
  model.eval()
  # Strip weight norm if running on TPU/XLA
  if hasattr(device, "type") and device.type in ["tpu", "xla"]:
    remove_weight_norm_recursive(model)

  eager_model = model
  if compile_flag:
    print("Compiling model...")
    model = torch.compile(model)

  # Generate 1 second of dummy audio at 16kHz
  dummy_audio = np.zeros(16000)

  inputs = feature_extractor(
      dummy_audio, sampling_rate=16000, return_tensors="pt"
  )
  inputs = cast_inputs_to_model_dtype(inputs, model, device)
  # Some models might need input_values or input_features key
  input_key = "input_values" if "input_values" in inputs else "input_features"
  inputs_tensor = inputs[input_key]
  forward_fn = lambda m, x: m(x)
  postprocess_fn = lambda out: f"Last hidden state shape: {out.last_hidden_state.shape}"
  duration, output_info, outputs, warmup_time = run_benchmark_pass(
      model, inputs_tensor, forward_fn, postprocess_fn, device=device
  )

  compile_time = warmup_time if compile_flag else None
  if compile_flag:
    print(f"INFO: Model compilation took {compile_time:.4f} seconds (measured via warmup).")

  mse_error = None
  if verify:
    mse_error = verify_output(
        outputs, eager_model, inputs_tensor, forward_fn, model_id, device, ref_dir
    )
  return duration, output_info, mse_error, compile_time, None
```
