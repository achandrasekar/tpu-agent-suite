# Diffusers Recipe
## Routing Rules

*   **Library**: `diffusers`

For Diffusers (Stable Diffusion / Image Generation):

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """Stable Diffusion benchmark logic."""
  diffusers = import_dependency("diffusers")
  import_dependency("accelerate")
  np = import_dependency("numpy")
  PIL = import_dependency("PIL")
  Image = getattr(PIL, "Image")

  # Apply local component patches
  try:
    pipeline_latent_diffusion = import_dependency("diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion")
    LDMBertModel = getattr(pipeline_latent_diffusion, "LDMBertModel")
    original_init = LDMBertModel.__init__
    def patched_init(self, *args, **kwargs):
      original_init(self, *args, **kwargs)
      if hasattr(self, "post_init") and not getattr(self, "_post_init_called", False):
        self.post_init()
        self._post_init_called = True
    LDMBertModel.__init__ = patched_init
  except Exception as e:
    print(f"Warning: Could not patch LDMBertModel: {e}")

  try:
    unet_2d = import_dependency("diffusers.models.unets.unet_2d")
    UNet2DModel = getattr(unet_2d, "UNet2DModel")
    original_unet_forward = UNet2DModel.forward
    def patched_unet_forward(self, sample, *args, **kwargs):
      try:
        weight_device = next(self.parameters()).device
        target_dtype = next(self.parameters()).dtype
        if weight_device.type in ["tpu", "xla", "cuda"]:
          if isinstance(sample, torch.Tensor) and sample.dtype != target_dtype:
            sample = sample.to(target_dtype)
      except Exception:
        pass
      return original_unet_forward(self, sample, *args, **kwargs)
    UNet2DModel.forward = patched_unet_forward
  except Exception as e:
    print(f"Warning: Could not patch UNet2DModel: {e}")

  try:
    vae = import_dependency("diffusers.models.autoencoders.vae")
    VectorQuantizer = getattr(vae, "VectorQuantizer")
    original_vq_forward = VectorQuantizer.forward
    def patched_vq_forward(self, z, *args, **kwargs):
      try:
        target_dtype = self.embedding.weight.dtype
        if isinstance(z, torch.Tensor) and z.dtype != target_dtype:
          z = z.to(target_dtype)
      except Exception:
        pass
      return original_vq_forward(self, z, *args, **kwargs)
    VectorQuantizer.forward = patched_vq_forward
  except Exception as e:
    print(f"Warning: Could not patch VectorQuantizer: {e}")

  huggingface_hub = import_dependency("huggingface_hub")
  hf_hub_download = getattr(huggingface_hub, "hf_hub_download")
  cache_dir = get_cache_dir()

  print(f"Resolving pipeline class for {model_id}...")
  pipeline_class_name = "StableDiffusionPipeline"
  try:
    index_file = hf_hub_download(repo_id=model_id, filename="model_index.json", cache_dir=cache_dir)
    with open(index_file, "r") as f:
      content = f.read()
      repaired_content = repair_json_string(content)
      json = import_dependency("json")
      index_data = json.loads(repaired_content)
      pipeline_class_name = index_data.get("_class_name", "StableDiffusionPipeline")
  except Exception as e:
    print(f"Warning: model_index.json routing failed: {e}. Falling back.")
    return benchmark_raw_fallback_model(
        model_id, device, cache_dir, verify=verify, ref_dir=ref_dir, compile_flag=compile_flag
    )

  pipeline_class = getattr(diffusers, pipeline_class_name, None)
  if pipeline_class is None:
    pipeline_class = getattr(diffusers, "StableDiffusionPipeline")

  kwargs = {}
  # Use bfloat16 for both CPU references and Accelerators to ensure math verification precision matches
  kwargs["torch_dtype"] = torch.bfloat16
  kwargs["cache_dir"] = cache_dir

  pipeline = pipeline_class.from_pretrained(model_id, **kwargs).to(device)
  pipeline.set_progress_bar_config(disable=True)

  eager_pipeline = pipeline
  if compile_flag:
    print("INFO: Compiling Diffusers components...")
    if hasattr(pipeline, "unet") and pipeline.unet is not None:
      print("INFO: Compiling UNet...")
      pipeline.unet = torch.compile(pipeline.unet)
    if hasattr(pipeline, "vae") and pipeline.vae is not None:
      print("INFO: Compiling VAE...")
      if hasattr(pipeline.vae, "decode"):
        pipeline.vae.decode = torch.compile(pipeline.vae.decode)
      else:
        pipeline.vae = torch.compile(pipeline.vae)

  is_text_conditional = "StableDiffusion" in pipeline_class_name or hasattr(pipeline, "tokenizer")
  is_upscaler = any(x in pipeline_class_name for x in ["SuperResolution", "Upscale", "Image2Image"])

  if is_text_conditional:
    forward_fn = lambda pipe, x: pipe(x, num_inference_steps=2, output_type="latent")
    inputs = "A simple red apple"
  elif is_upscaler:
    inputs = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    forward_fn = lambda pipe, x: pipe(image=x, num_inference_steps=2, output_type="latent")
  else:
    inputs = None
    forward_fn = lambda pipe, x: pipe(batch_size=1, num_inference_steps=2, output_type="latent")

  postprocess_fn = lambda out: f"diffusers {pipeline_class_name} execution completed."

  duration, output_info, outputs, warmup_time = run_benchmark_pass(
      pipeline, inputs, forward_fn, postprocess_fn, device=device
  )

  compile_time = warmup_time if compile_flag else None
  if compile_flag:
    print(f"INFO: Model compilation took {compile_time:.4f} seconds (measured via warmup).")

  mse_error = None
  if verify:
    mse_error = verify_output(
        outputs, eager_pipeline, inputs, forward_fn, model_id, device, ref_dir
    )

  return duration, output_info, mse_error, compile_time, None


def benchmark_raw_fallback_model(
    model_id: str,
    device: torch.device,
    cache_dir: str,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """Raw fallback loaders evaluate logic."""
  transformers = import_dependency("transformers")
  AutoConfig = getattr(transformers, "AutoConfig")
  AutoModel = getattr(transformers, "AutoModel")

  print(f"Loading raw fallback model {model_id}...")
  config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
  model = AutoModel.from_pretrained(model_id, config=config, cache_dir=cache_dir, trust_remote_code=True).to(device)
  model.eval()

  eager_model = model
  if compile_flag:
    print("Compiling fallback model...")
    model = torch.compile(model)

  inputs = torch.ones((1, 10), dtype=torch.int64).to(device)

  # Check if model takes long inputs
  try:
    with torch.no_grad():
      _ = model(inputs)
  except Exception as e:
    print(f"Warning: Long inputs (int64) failed: {e}. Falling back to float inputs.")
    inputs = torch.ones((1, 10), dtype=torch.float32).to(device)

  forward_fn = lambda m, x: m(x)
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
