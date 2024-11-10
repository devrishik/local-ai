# DSPy Prompt Optimization with Small Language Models

This project implements prompt optimization using DSPy's MiProv2 optimizer with Small Language Models (SLMs). Currently supported models:
- Llama-3.2-1.3B
- Mistral-3B

The system automatically detects and utilizes the best available hardware acceleration (CUDA, Metal, CPU) for optimal performance.

## 🌟 Features

- Efficient prompt optimization using DSPy's MiProv2
- Hardware acceleration priority:
  1. CUDA (NVIDIA GPUs)
  2. Metal (Apple Silicon)
  3. CPU (fallback)
- Memory-optimized model loading
- Automatic mixed precision when available
- Runtime hardware monitoring

## 🛠️ Prerequisites

- Python 3.8 or higher
- One of:
  - NVIDIA GPU with CUDA support
  - Apple Silicon Mac
  - Modern CPU (fallback)
- [Rye](https://rye-up.com/) package manager
- Hugging Face account for model access

## 🚀 Installation

1. **Install Rye** if you haven't already:
   ```bash
   curl -sSf https://rye-up.com/get | bash
   ```

2. **Clone the repository**:
   ```bash
   git clone git@github.com:devrishik/local-ai.git
   cd dspy-prompt-optimizer
   ```

3. **Install dependencies**:
   ```bash
   rye sync --update-all
   ```

4. **Login to Hugging Face**:
   ```bash
   huggingface-cli login
   ```

## 💻 Usage

Run the optimization script:
```bash
rye run optimize
```

The script will:
1. Detect available hardware acceleration
2. Select and load the appropriate SLM
3. Optimize prompts using MiProv2
4. Display hardware utilization statistics

## 📊 Example Hardware Detection Output

```bash
Checking available hardware...
✓ CUDA detected: NVIDIA RTX 3080 (10GB)
✗ Metal acceleration not available
✓ CPU cores available: 12

Selected acceleration: CUDA
Loading Llama-3.2-1.3B...
Memory Allocated: 1.3GB VRAM
```


## 🗃️ Project Structure

dspy-prompt-optimizer/
├── README.md
├── pyproject.toml
├── src/
│   └── local_ai/
│       ├── __init__.py
│       ├── main.py
│       └── ml/
│           └── models.py
```

## 💾 Hardware Requirements

### NVIDIA GPU
- Minimum 5GB VRAM for Mistral-3B
- CUDA 11.7 or higher

### Apple Silicon
- Minimum 8GB unified memory
- macOS 12.0 or higher

### CPU Only
- Minimum 16GB RAM
- AVX2 instruction set support recommended

## 🔍 Troubleshooting

### CUDA Issues
install pytorch with cuda support: https://telin.ugent.be/telin-docs/windows/pytorch/
```bash
# Check CUDA availability
python -c "import torch; print(torch.rand(2,3).cuda())"
```

### Memory Issues
```python
# Monitor memory usage
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
print(f"System Memory: {psutil.Process().memory_info().rss/1024**3:.2f}GB")
```

### Common Solutions
1. **Out of Memory**:
   - Try the smaller Llama-3.2-1.3B model
   - Reduce batch size
   - Enable gradient checkpointing

2. **Model Loading Errors**:
   - Verify Hugging Face login
   - Check internet connection
   - Confirm hardware compatibility

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [DSPy](https://dspy-docs.vercel.app/) team for the optimization framework
- [Llama](https://ai.meta.com/llama/) team at Meta AI
- [Mistral AI](https://mistral.ai/) team
- [Hugging Face](https://huggingface.co/) for model hosting

## 📧 Contact

For questions or support, open an issue in the repository.