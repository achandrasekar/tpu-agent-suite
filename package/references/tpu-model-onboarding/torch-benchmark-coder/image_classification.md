# Image Classification Recipe
## Routing Rules
Apply this recipe if:

*   `pipeline_tag` is `image-classification`

For standard vision classification models (e.g., ViT, DeiT, ConvNeXt) loaded via
transformers:

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """Image classification benchmark forward logic."""
  transformers = import_dependency("transformers")
  AutoModelForImageClassification = getattr(transformers, "AutoModelForImageClassification")
  AutoImageProcessor = getattr(transformers, "AutoImageProcessor")

  print(f"Loading {model_id}...")
  model, image_processor = load_model_and_processor(
      model_id, device, AutoModelForImageClassification, AutoImageProcessor, use_bfloat16=True
  )
  model.eval()

  eager_model = model
  if compile_flag:
    print("Compiling model...")
    model = torch.compile(model)

  # Get input resolution from image processor config, default to 224
  size = 224
  if hasattr(image_processor, "size") and "height" in image_processor.size:
    size = image_processor.size["height"]
  elif hasattr(image_processor, "crop_size") and "height" in image_processor.crop_size:
    size = image_processor.crop_size["height"]
  # Generate dummy image matching expected size
  dummy_image = torch.zeros((1, 3, size, size))
  if hasattr(device, "type") and device.type != "cpu":
    dummy_image = dummy_image.to(device).to(torch.bfloat16)
  else:
    dummy_image = dummy_image.to(device)
  forward_fn = lambda m, x: m(x)
  postprocess_fn = lambda out: f"Logits shape: {out.logits.shape}"
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
