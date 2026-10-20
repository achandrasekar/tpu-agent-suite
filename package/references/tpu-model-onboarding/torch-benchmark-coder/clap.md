# CLAP Recipe
## Routing Rules
Apply this recipe if:

*   `model_id` contains "clap" (case-insensitive)

For CLAP models follow below approach to benchmark (e.g. laion/clap-htsat-fused,
  laion/larger_clap_music_and_speech, laion/larger_clap_general .etc):

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    library: str = "transformers",
    pipeline_tag: str = "text-generation",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """CLAP benchmark forward logic."""
  transformers = import_dependency("transformers")
  np = import_dependency("numpy")
  ClapModel = getattr(transformers, "ClapModel")
  AutoProcessor = getattr(transformers, "AutoProcessor")

  print(f"Loading {model_id}...")
  model, processor = load_model_and_processor(
      model_id, device, ClapModel, AutoProcessor, use_bfloat16=True
  )
  model.eval()

  eager_model = model
  if compile_flag:
    print("Compiling model...")
    model = torch.compile(model)

  # 10s audio at 48kHz
  audio = np.random.rand(480000)
  inputs = processor(audio=audio, text=["a sound of a dog", "a sound of a cat"], sampling_rate=48000, return_tensors="pt")
  inputs = cast_inputs_to_model_dtype(inputs, model, device)

  verify_inputs_bounds(model, inputs)

  forward_fn = lambda m, x: m(**x)
  postprocess_fn = lambda out: f"Logits per audio shape: {out.logits_per_audio.shape}"

  duration, output_info, outputs, warmup_time = run_benchmark_pass(
      model, inputs, forward_fn, postprocess_fn, device=device
  )

  compile_time = warmup_time if compile_flag else None
  if compile_flag:
    print(f"INFO: Model compilation took {compile_time:.4f} seconds (measured via warmup).")

  mse_error = None
  if verify:
    mse_error = verify_output(
        outputs, eager_model, inputs, forward_fn, model_id, device, ref_dir
    )

  return duration, output_info, mse_error, compile_time, None
```
