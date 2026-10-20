# Raw Fallback Recipe
## Routing Rules
Apply this recipe as a default fallback when no other specific rules match.

For raw fallback models when no specific recipe matches:

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """Raw fallback loaders evaluate logic."""
  transformers = import_dependency("transformers")
  AutoConfig = getattr(transformers, "AutoConfig")
  AutoModel = getattr(transformers, "AutoModel")
  cache_dir = get_cache_dir()

  print(f"Loading raw fallback model {model_id}...")
  config = load_config_with_fallback(model_id, cache_dir)
  model = AutoModel.from_pretrained(model_id, config=config, cache_dir=cache_dir, trust_remote_code=True).to(device)
  model.eval()

  eager_model = model
  if compile_flag:
    print("Compiling fallback model...")
    model = torch.compile(model)

  inputs = get_dummy_inputs_for_config(config, device)

  # Check if model takes generated inputs
  def test_forward(m, x):
    if isinstance(x, dict):
      return m(**x)
    return m(x)

  try:
    with torch.no_grad():
      _ = test_forward(model, inputs)
  except Exception as e:
    print(f"Warning: Default dummy inputs failed: {e}. Attempting fallback...")
    if isinstance(inputs, dict):
      pass
    elif isinstance(inputs, torch.Tensor) and inputs.ndim == 4:
      inputs = inputs.to(torch.float32)
    elif isinstance(inputs, torch.Tensor) and inputs.dtype == torch.int64:
      inputs = torch.ones((1, 10), dtype=torch.float32).to(device)

  forward_fn = lambda m, x: m(**x) if isinstance(x, dict) else m(x)
  postprocess_fn = lambda out: "Fallback raw model execution completed."

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
