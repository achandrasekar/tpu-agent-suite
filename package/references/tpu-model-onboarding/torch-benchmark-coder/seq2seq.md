# Seq2Seq / Encoder-Decoder Recipe
## Routing Rules
Apply this recipe if:

*   `pipeline_tag` is in [`text2text-generation`, `translation`, `summarization`]

For Seq2Seq models (e.g., T5, Flan-T5, NLLB, BART) that require encoder-decoder
attention and generate outputs:

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """Seq2Seq benchmark forward logic."""
  transformers = import_dependency("transformers")
  AutoModelForSeq2SeqLM = getattr(transformers, "AutoModelForSeq2SeqLM")
  AutoTokenizer = getattr(transformers, "AutoTokenizer")

  print(f"Loading {model_id}...")
  model, tokenizer = load_model_and_processor(
      model_id, device, AutoModelForSeq2SeqLM, AutoTokenizer, use_bfloat16=True
  )
  model.eval()

  eager_model = model
  if compile_flag:
    print("Compiling model...")
    model = torch.compile(model)

  prompt = "translate English to German: Hello, how are you?"
  inputs = tokenizer(prompt, return_tensors="pt")
  inputs = cast_inputs_to_model_dtype(inputs, model, device)
  # Use generate to handle decoder_input_ids and autoregressive loop automatically
  forward_fn = lambda m, x: m.generate(**x, min_new_tokens=50, max_new_tokens=50, do_sample=False, temperature=0.0)
  postprocess_fn = lambda out: tokenizer.decode(
      out.cpu()[0], skip_special_tokens=True
  )
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
