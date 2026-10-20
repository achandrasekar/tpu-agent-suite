# Timm Recipe
## Routing Rules

*   **Library**: `timm`

For Timm (Torch Image Models):

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """Timm loaders evaluate logic."""
  timm = import_dependency("timm")

  print(f"Loading timm model {model_id}...")
  if (hasattr(device, "type") and device.type == "cpu") or str(device) == "cpu":
    model = timm.create_model(model_id, pretrained=True).to(device)
  else:
    model = timm.create_model(model_id, pretrained=True).to(device).to(torch.bfloat16)
  model.eval()

  eager_model = model
  if compile_flag:
    print("Compiling model...")
    model = torch.compile(model)

  input_size = (3, 224, 224)
  if hasattr(model, "default_cfg") and model.default_cfg.get("input_size"):
    input_size = model.default_cfg["input_size"]

  dummy_image = torch.zeros((1, *input_size))
  if not ((hasattr(device, "type") and device.type == "cpu") or str(device) == "cpu"):
    dummy_image = dummy_image.to(device).to(torch.bfloat16)
  else:
    dummy_image = dummy_image.to(device)

  forward_fn = lambda m, x: m(x)
  postprocess_fn = lambda out: f"timm Logits shape: {out.shape}"

  duration, output_info, outputs, warmup_time = run_benchmark_pass(
      model, dummy_image, forward_fn, postprocess_fn, device=device
  )

  compile_time = warmup_time if compile_flag else None
  if compile_flag:
    print(f"INFO: Model compilation took {compile_time:.4f} seconds (measured via warmup).")

  mse_error = None
  if verify:
    mse_error = verify_output(
        outputs, eager_model, dummy_image, forward_fn, model_id, device, ref_dir
    )

  return duration, output_info, mse_error, compile_time, None
```
