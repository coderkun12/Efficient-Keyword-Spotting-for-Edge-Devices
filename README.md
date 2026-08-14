# Efficient Keyword Spotting for Edge Devices
 
Training a small speech-command classifier and compressing it with **pruning** and **quantization** to make it viable for always-on, resource-constrained deployment (the kind of setup behind wake-word detection like "Hey Siri" / "Hey Google").
 
> **TL;DR:** Wake-word detection has to run 24/7 on tiny, low-power hardware. That constraint isn't optional — it's the entire reason model compression matters here. This project trains a keyword-spotting CNN, then prunes and quantizes it to show how much smaller and faster it can get while holding onto most of its accuracy.
 
---
 
## 🎯 Project Goals
 
- Train a baseline CNN keyword-spotting model on the [Google Speech Commands dataset](https://arxiv.org/abs/1804.03209).
- Apply **unstructured** and **structured pruning** and measure the accuracy/size/speed tradeoffs.
- Apply **post-training quantization (PTQ)** and **quantization-aware training (QAT)** and compare accuracy recovery.
- Combine pruning + quantization into a single compression pipeline.
- Benchmark model size, parameter count, and CPU inference latency at every stage.
- *(Stretch goal)* Export the final model to ONNX/TFLite and run a live browser or Raspberry Pi demo.
---
 
## 🧠 Why This Project
 
Most compression demos prune a model just to prove it *can* be pruned. Keyword spotting is different — the deployment constraint is real: an always-listening microcontroller-class chip has a tiny memory and power budget, so a 1MB+ FP32 model simply isn't an option. This project treats compression as a requirement, not an exercise, and measures results against that lens.
 
---
 
## 📊 Dataset
 
**[Google Speech Commands v0.02](https://arxiv.org/abs/1804.03209)** (Warden, 2018)
- ~105,000 one-second audio clips
- 35 spoken word classes (e.g. "yes", "no", "stop", "go", digits)
- Loaded via `torchaudio.datasets.SPEECHCOMMANDS`
For this project, a subset of **10–12 keyword classes** is used initially to keep iteration fast, with the option to scale to the full 35-class problem later.
 
---
 
## 🏗️ Model
 
A small CNN operating on **mel-spectrogram** representations of the audio (rather than raw waveform), similar in spirit to compact keyword-spotting architectures like DS-CNN described in ["Hello Edge: Keyword Spotting on Microcontrollers"](https://arxiv.org/abs/1711.07128) (Zhang et al., 2017).
 
```
Input (mel-spectrogram)
   → Conv2D + ReLU + BatchNorm
   → Conv2D + ReLU + BatchNorm
   → Global Average Pooling
   → Fully Connected
   → Softmax (N classes)
```
 
*(Exact architecture and layer sizes documented in `model.py`.)*
 
---
 
## 🛠️ Tech Stack
 
| Component | Tool |
|---|---|
| Framework | PyTorch |
| Audio processing | torchaudio |
| Pruning | `torch.nn.utils.prune` |
| Quantization | `torch.quantization` (PTQ + QAT) |
| Export | ONNX / TFLite |
| Environment | Local (RTX 2050, 4GB VRAM) — no cloud GPU required |
 
---
 
## 📁 Repository Structure
 
```
├── data/                  # Dataset download/cache location (gitignored)
├── src/
│   ├── dataset.py         # Speech Commands loading + preprocessing
|   ├── dataExtract.py     # Runs the process of downloading the data set.
│   ├── model.py           # CNN architecture
│   ├── train.py           # Baseline training loop
│   ├── prune.py           # Pruning experiments (unstructured + structured)
│   ├── quantize.py        # PTQ + QAT pipelines
│   └── benchmark.py       # Size / latency / accuracy measurement
├── notebooks/             # Exploratory analysis, plots
├── results/               # Saved metrics, plots, model checkpoints
├── requirements.txt
└── README.md
```
 
---
 
## 🚀 Getting Started
 
```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
 
# Install dependencies
pip install -r requirements.txt
 
# Train the baseline model
python src/train.py
 
# Run pruning experiments
python src/prune.py
 
# Run quantization experiments
python src/quantize.py
 
# Benchmark all model variants
python src/benchmark.py
```
 
---
 
## 📈 Results
 
*(To be filled in as experiments are completed.)*
 
| Model Variant | Size | Accuracy | CPU Latency | Params |
|---|---|---|---|---|
| Baseline (FP32) | — | — | — | — |
| Pruned (50%) | — | — | — | — |
| Pruned (90%) | — | — | — | — |
| PTQ (INT8) | — | — | — | — |
| QAT (INT8) | — | — | — | — |
| Pruned + Quantized | — | — | — | — |
 
### Accuracy vs. Compression
*(Plot to be added)*
 
### Key Findings
*(Notes on what actually worked, what surprised me, and why — e.g. whether unstructured pruning gave a real speedup, how much QAT recovered vs PTQ, etc.)*
 
---
 
## 🔭 Future Work
 
- [ ] Scale to the full 35-class problem
- [ ] Structured (channel-level) pruning for real inference speedup
- [ ] Deploy quantized model to Raspberry Pi and measure real-world latency/power
- [ ] Live browser demo using ONNX.js / TF.js
- [ ] Knowledge distillation as an alternative/complementary compression approach
---
 
## 📚 References
 
- Warden, P. (2018). [*Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*](https://arxiv.org/abs/1804.03209)
- Zhang, Y. et al. (2017). [*Hello Edge: Keyword Spotting on Microcontrollers*](https://arxiv.org/abs/1711.07128)
- Frankle, J. & Carbin, M. (2019). [*The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*](https://arxiv.org/abs/1803.03635)
- [PyTorch Speech Command Classification Tutorial](https://docs.pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html)
- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
---
 
## 📝 License
 
MIT License — see [`LICENSE`](LICENSE) for details.
