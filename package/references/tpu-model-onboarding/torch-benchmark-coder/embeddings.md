# Embeddings / Text Encoders Recipe
## Routing Rules
Apply this recipe if:

*   `pipeline_tag` is in [`feature-extraction`, `sentence-similarity`, `fill-mask`, `question-answering`, `text-classification`]

For feature vectors & embeddings:

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """Feature vectors embedding benchmark forward logic."""
  transformers = import_dependency("transformers")
  AutoModel = getattr(transformers, "AutoModel")
  AutoTokenizer = getattr(transformers, "AutoTokenizer")

  print(f"Loading {model_id}...")
  model, tokenizer = load_model_and_processor(
      model_id, device, AutoModel, AutoTokenizer, use_bfloat16=False
  )
  model.eval()

  eager_model = model
  if compile_flag:
    print("Compiling model...")
    model = torch.compile(model)

  inputs = tokenizer("Hello, what can you help me with?", return_tensors="pt")
  inputs = cast_inputs_to_model_dtype(inputs, model, device)

  forward_fn = lambda m, x: m(**x)
  postprocess_fn = lambda out: f"Last hidden state shape: {out.last_hidden_state.shape}"

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
