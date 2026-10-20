# Vision Segmentation / Object Detection Recipe
## Routing Rules
Apply this recipe if:

*   `pipeline_tag` is in [`image-segmentation`, `object-detection`, `mask-generation`]

For vision segmentation and object detection:

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """Segmentation benchmark forward logic."""
  transformers = import_dependency("transformers")
  from transformers import AutoModel
  AutoModel = getattr(transformers, "AutoModel")

  print(f"Loading {model_id}...")
  model = load_model_and_processor(
      model_id, device, AutoModel, use_bfloat16=False
  )
  model.eval()

  eager_model = model
  if compile_flag:
    print("Compiling model...")
    model = torch.compile(model)

  dummy_image = torch.zeros((1, 3, 1024, 1024))
  dummy_image = cast_inputs_to_model_dtype(dummy_image, model, device)

  forward_fn = lambda m, x: m(x)
  def postprocess_fn(outputs):
    if isinstance(outputs, (list, tuple)):
      return f"Tuple output, first element shape: {outputs[0].shape}"
    elif hasattr(outputs, "logits"):
      return f"Logits shape: {outputs.logits.shape}"
    elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
      return f"Last hidden state shape: {outputs.last_hidden_state.shape}"
    else:
      return f"Output shape: {outputs.shape}"

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
