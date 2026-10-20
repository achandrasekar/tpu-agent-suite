# Vision Language Model (VLM) Recipe
## Routing Rules

Apply this recipe if:

*   `pipeline_tag` is in [`text-image-to-text`, `visual-question-answering`, `any-to-any`]
*   **OR** `model_id` contains any of [`llava`, `vlm`, `internvl`, `qwen2.5-vl`, `qwen2-vl`, `smolvlm`, `paligemma`, `idefics`] (case-insensitive)

For text generation LLMs (Llama, Gemma, GPT, Qwen, etc.):

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    library: str = "transformers",
    pipeline_tag: str = "image-text-to-text",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """Vision-Language Model (VLM) benchmark forward logic."""
  np = import_dependency("numpy")
  PIL = import_dependency("PIL")
  Image = getattr(PIL, "Image")
  transformers = import_dependency("transformers")

  AutoProcessor = getattr(transformers, "AutoProcessor")
  # Fallback gracefully between AutoModelForImageTextToText and AutoModel
  try:
    AutoModelForVLM = getattr(transformers, "AutoModelForImageTextToText")
  except AttributeError:
    try:
      AutoModelForVLM = getattr(transformers, "AutoModelForVision2Seq")
    except AttributeError:
      AutoModelForVLM = getattr(transformers, "AutoModel")

  print(f"Loading VLM model: {model_id}...")
  model, processor = load_model_and_processor(
      model_id, device, AutoModelForVLM, AutoProcessor, use_bfloat16=True
  )
  model.eval()

  eager_model = model
  # Apply torch.compile on GPU/CPU only; PyTorch/XLA compiles on-the-fly during warmup pass on TPU
  if compile_flag and device.type != "tpu":
    print("Compiling model with torch.compile...")
    model = torch.compile(model)

  # Generate synthetic in-memory image to avoid live HTTP network dependencies
  dummy_array = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
  image = Image.fromarray(dummy_array)

  prompt = "Describe this image in detail."

  # Format prompt with Chat Template if available
  if hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
      input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    except Exception:
      input_text = prompt
  else:
    input_text = prompt

  # Preprocess image and text inputs
  try:
    inputs = processor(text=input_text, images=image, return_tensors="pt")
  except Exception:
    inputs = processor(images=image, return_tensors="pt")

  # Move input tensors to target device and precision (bfloat16/float32)
  inputs = cast_inputs_to_model_dtype(inputs, model, device)

  verify_inputs_bounds(model, inputs)

  # Define forward pass (using generate if supported, otherwise standard forward)
  if hasattr(model, "generate"):
    forward_fn = lambda m, x: m.generate(**x, min_new_tokens=30, max_new_tokens=30, do_sample=False, temperature=0.0)
    postprocess_fn = lambda out: processor.batch_decode(out.cpu(), skip_special_tokens=True)[0]
  else:
    forward_fn = lambda m, x: m(**x)
    postprocess_fn = lambda out: f"VLM output logits shape: {extract_raw_tensor(out).shape}"

  # Run warmup and steady-state timing benchmark
  duration, output_info, outputs, warmup_time = run_benchmark_pass(
      model, inputs, forward_fn, postprocess_fn, device=device
  )

  compile_time = warmup_time if compile_flag else None
  if compile_flag:
    print(f"INFO: Model graph compilation took {compile_time:.4f} seconds (measured via warmup).")

  # Calculate MSE against baseline reference tensor
  mse_error = None
  ppl_stats = None
  if verify:
    # 1. KV-Cache Generation Check
    mse_error = verify_output(
        outputs, eager_model, inputs, forward_fn, model_id, device, ref_dir
    )
    
    # 2. Core Math Check
    device_ppl, rel_deg = calculate_perplexity(eager_model, inputs)
    ppl_stats = {"device_ppl": device_ppl, "relative_degradation": rel_deg}

  return duration, output_info, mse_error, compile_time, ppl_stats
```
