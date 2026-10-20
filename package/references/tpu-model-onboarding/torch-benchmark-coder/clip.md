# CLIP Recipe
## Routing Rules
Apply this recipe if:

*   `model_id` contains "clip" (case-insensitive)
*   **OR** `pipeline_tag` is in [`zero-shot-image-classification`, `image-to-text`] **AND** `model_id` contains "clip"

For CLIP models follow below approach to benchmark:

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """CLIP benchmark forward logic."""
  PIL = import_dependency("PIL")
  Image = getattr(PIL, "Image")
  requests = import_dependency("requests")
  transformers = import_dependency("transformers")
  CLIPProcessor = getattr(transformers, "CLIPProcessor")
  CLIPModel = getattr(transformers, "CLIPModel")

  print(f"Loading {model_id}...")
  model, processor = load_model_and_processor(
      model_id, device, CLIPModel, CLIPProcessor, use_bfloat16=True
  )
  model.eval()

  eager_model = model
  if compile_flag:
    print("Compiling model...")
    model = torch.compile(model)

  url = "http://images.cocodataset.org/val2017/000000039769.jpg"
  image = Image.open(requests.get(url, stream=True).raw)


  prompt = "a photo of a cat"
  if processor.chat_template:
    messages = [{"role": "user", "content": prompt}]
    input_text = processor.apply_chat_template(messages, tokenize=False)
  else:
    input_text = prompt

  inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
  inputs = cast_inputs_to_model_dtype(inputs, model, device)

  verify_inputs_bounds(model, inputs)

  forward_fn = lambda m, x: m(**x)
  postprocess_fn = lambda out: f"Logits per image shape: {out.logits_per_image.shape}"

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
