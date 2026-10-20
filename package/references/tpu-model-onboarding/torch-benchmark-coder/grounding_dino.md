# Grounding DINO Recipe
## Routing Rules
Apply this recipe if:

*   `model_id` contains "grounding-dino" or "groundingdino" (case-insensitive)

For Grounding DINO object detection models that require both image and text
query inputs:

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """Grounding DINO benchmark forward logic."""
  PIL = import_dependency("PIL")
  Image = getattr(PIL, "Image")
  transformers = import_dependency("transformers")
  AutoModelForZeroShotObjectDetection = getattr(transformers, "AutoModelForZeroShotObjectDetection")
  AutoProcessor = getattr(transformers, "AutoProcessor")

  print(f"Loading {model_id}...")
  model, processor = load_model_and_processor(
      model_id, device, AutoModelForZeroShotObjectDetection, AutoProcessor, use_bfloat16=True
  )
  model.eval()

  eager_model = model
  if compile_flag:
    print("Compiling model...")
    model = torch.compile(model)

  # Generate dummy image
  dummy_image = Image.new("RGB", (800, 800))

  # Prepare inputs with text prompt and image
  inputs = processor(
      images=dummy_image,
      text="detect the cat . detect the dog .",
      return_tensors="pt"
  )
  inputs = cast_inputs_to_model_dtype(inputs, model, device)
  forward_fn = lambda m, x: m(**x)
  postprocess_fn = (
      lambda out: f"Pred logits shape: {out.pred_logits.shape}"
  )
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
