# Causal LM Recipe
## Routing Rules
Apply this recipe if:

*   `pipeline_tag` is in [`text-generation`, `conversational`]
*   **AND** none of the more specific rules (like Whisper) match

For text generation LLMs (Llama, Gemma, GPT, Qwen, etc.):

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """CausalLM benchmark forward logic."""
  transformers = import_dependency("transformers")
  AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
  AutoTokenizer = getattr(transformers, "AutoTokenizer")
  print(f"Loading {model_id}...")
  model, tokenizer = load_model_and_processor(
      model_id, device, AutoModelForCausalLM, AutoTokenizer, use_bfloat16=True
  )
  model.eval()

  eager_model = model
  if compile_flag:
    print("Compiling model...")
    model = torch.compile(model)

  prompt = "Hello, what can you help me with?"

  if tokenizer.chat_template:
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
  else:
    input_text = prompt

  inputs = tokenizer(input_text, return_tensors="pt")
  inputs = cast_inputs_to_model_dtype(inputs, model, device)

  verify_inputs_bounds(model, inputs)

  forward_fn = lambda m, x: m.generate(**x, min_new_tokens=50, max_new_tokens=50, do_sample=False, temperature=0.0)
  postprocess_fn = lambda out: tokenizer.decode(out.cpu()[0], skip_special_tokens=True)

  duration, output_info, outputs, warmup_time = run_benchmark_pass(
      model, inputs, forward_fn, postprocess_fn, device=device
  )

  compile_time = warmup_time if compile_flag else None
  if compile_flag:
    print(f"INFO: Model compilation took {compile_time:.4f} seconds (measured via warmup).")

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
