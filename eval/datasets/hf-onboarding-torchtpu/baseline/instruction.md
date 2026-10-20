I would like to onboard a model from HF to TPU using `torch_tpu`, the native PyTorch TPU backend. I
want to know what it costs to serve before we commit hardware to it. The model information is in
your env. `env | grep HF_MODEL`

- **Target Slice**: Recommended TP=1, Context Window=32768

---

If cheap analysis exists so that a benchmark is not spent answering a question arithmetic could have
answered.
