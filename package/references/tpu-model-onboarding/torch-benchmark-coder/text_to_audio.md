# Text to Audio Recipe
## Routing Rules
Apply this recipe if:

*   `pipeline_tag` is `text-to-audio`

For text-to-audio models (e.g. MusicGen) loaded via transformers:

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """Text-to-Audio models evaluate logic (e.g. Musicgen)."""
  transformers = import_dependency("transformers")
  AutoConfig = getattr(transformers, "AutoConfig")
  AutoModelForTextToWaveform = getattr(transformers, "AutoModelForTextToWaveform")
  cache_dir = get_cache_dir()

  print(f"Loading text-to-audio model {model_id}...")
  config = load_config_with_fallback(model_id, cache_dir)
  model = AutoModelForTextToWaveform.from_pretrained(
      model_id, config=config, cache_dir=cache_dir, trust_remote_code=True
  ).to(device)
  model.eval()

  eager_model = model
  if compile_flag:
    print("Compiling text-to-audio model...")
    model = torch.compile(model)

  inputs = get_dummy_inputs_for_config(config, device)

  def test_forward(m, x):
    if isinstance(x, dict):
      return m(**x)
    return m(x)

  postprocess_fn = lambda out: f"Logits shape: {out.logits.shape}"

  duration, output_info, outputs, warmup_time = run_benchmark_pass(
      model, inputs, test_forward, postprocess_fn, device=device
  )

  compile_time = warmup_time if compile_flag else None
  if compile_flag:
    print(
        f"INFO: Model compilation took {compile_time:.4f} seconds (measured via"
        f" warmup)."
    )

  mse_error = None
  if verify:
    mse_error = verify_output(
        outputs, eager_model, inputs, test_forward, model_id, device, ref_dir
    )

  return duration, output_info, mse_error, compile_time, None
```
