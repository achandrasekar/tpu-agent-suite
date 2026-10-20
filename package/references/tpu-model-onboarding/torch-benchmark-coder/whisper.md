# Whisper Recipe
## Routing Rules
Apply this recipe if:

*   `model_id` contains "whisper" (case-insensitive)
*   **OR** `pipeline_tag` is `automatic-speech-recognition` **AND** `model_id` contains "whisper"

Consider using this recipe for Whisper / speech recognition models.

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """Whisper benchmark forward logic."""
  np = import_dependency("numpy")
  transformers = import_dependency("transformers")
  AutoModelForSpeechSeq2Seq = getattr(transformers, "AutoModelForSpeechSeq2Seq")
  AutoProcessor = getattr(transformers, "AutoProcessor")

  print(f"Loading {model_id}...")
  model, processor = load_model_and_processor(
      model_id, device, AutoModelForSpeechSeq2Seq, AutoProcessor, use_bfloat16=True
  )
  model.eval()

  eager_model = model
  if compile_flag:
    print("Compiling model...")
    model = torch.compile(model)

  dummy_audio = np.zeros(16000)
  inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
  inputs = cast_inputs_to_model_dtype(inputs, model, device)
  input_features = inputs["input_features"]

  forward_fn = lambda m, x: m.generate(x, max_new_tokens=50)
  postprocess_fn = lambda out: processor.batch_decode(out.cpu(), skip_special_tokens=True)[0]

  duration, output_info, outputs, warmup_time = run_benchmark_pass(
      model, input_features, forward_fn, postprocess_fn, device=device
  )

  compile_time = warmup_time if compile_flag else None
  if compile_flag:
    print(f"INFO: Model compilation took {compile_time:.4f} seconds (measured via warmup).")

  mse_error = None
  ppl_stats = None
  if verify:
    # 1. KV-Cache Generation Check
    mse_error = verify_output(
        outputs, eager_model, input_features, forward_fn, model_id, device, ref_dir
    )

  return duration, output_info, mse_error, compile_time, ppl_stats
```

