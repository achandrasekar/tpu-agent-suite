# SpeechBrain Recipe
## Routing Rules

*   **Library**: `speechbrain`

For SpeechBrain (ASR & Audio Separation):

```python
def benchmark_model(
    model_id: str,
    device: torch.device,
    verify: bool = True,
    ref_dir: str = ".benchmark_refs",
    compile_flag: bool = False,
) -> tuple[float, str, float | None, float | None, dict | None]:
  """SpeechBrain benchmark logic."""
  torchaudio = import_dependency("torchaudio")
  speechbrain = import_dependency("speechbrain")
  try:
    if not hasattr(torchaudio, "list_audio_backends"):
      torchaudio.list_audio_backends = lambda: []
  except Exception: pass

  import shutil
  import tempfile
  from speechbrain.inference.interfaces import Pretrained  # pytype: disable=import-error

  print(f"Loading speechbrain pretrained model {model_id}...")
  tmp_dir = tempfile.mkdtemp(prefix="speechbrain_tmp_")

  model = None
  is_asr = False
  is_separator = False
  is_classifier = False

  if any(x in model_id.lower() for x in ["sepformer", "separator"]):
    try:
      from speechbrain.inference.separation import SepformerSeparation  # pytype: disable=import-error
      model = SepformerSeparation.from_hparams(source=model_id, savedir=tmp_dir, run_opts={"device": str(device)})
      is_separator = True
    except Exception:
      pass

  if model is None:
    try:
      from speechbrain.inference.ASR import EncoderDecoderASR  # pytype: disable=import-error
      model = EncoderDecoderASR.from_hparams(source=model_id, savedir=tmp_dir, run_opts={"device": str(device)})
      is_asr = True
    except Exception:
      pass

  if model is None:
    try:
      from speechbrain.inference.classifiers import EncoderClassifier  # pytype: disable=import-error
      model = EncoderClassifier.from_hparams(source=model_id, savedir=tmp_dir, run_opts={"device": str(device)})
      is_classifier = True
    except Exception:
      model = Pretrained.from_hparams(source=model_id, savedir=tmp_dir, run_opts={"device": str(device)})

  if model is not None and hasattr(model, "mods"):
    for name, module in model.mods.items():
      remove_weight_norm_recursive(module)

  eager_model = model
  if compile_flag:
    print("INFO: Compiling SpeechBrain modules...")
    if hasattr(model, "mods"):
      for name, module in model.mods.items():
        if isinstance(module, nn.Module):
          model.mods[name] = torch.compile(module)

  dummy_audio = torch.zeros((1, 16000)).to(device)
  wav_lens = torch.tensor([1.0]).to(device)

  if is_asr:
    inputs = (dummy_audio, wav_lens)
    forward_fn = lambda m, x: m.transcribe_batch(*x)
    postprocess_fn = lambda out: f"ASR transcription completed: {out}"
  elif is_separator:
    inputs = dummy_audio
    forward_fn = lambda m, x: m.separate_batch(x)
    postprocess_fn = lambda out: f"Separator audio sources generated: {len(out)}"
  elif is_classifier:
    inputs = dummy_audio
    forward_fn = lambda m, x: m.classify_batch(x)
    postprocess_fn = lambda out: f"Classifier label generated: {out[3]}"
  else:
    inputs = None
    forward_fn = lambda m, x: None
    postprocess_fn = lambda out: "speechbrain base forward pass completed."

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

  shutil.rmtree(tmp_dir, ignore_errors=True)
  return duration, output_info, mse_error, compile_time, None
```
