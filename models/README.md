# Models

The fine-tuned FunctionGemma 270M GGUF (compact tool-call format,
Q4_K_M quantization, 253 MB) is hosted publicly on HuggingFace.

From the `functiongemma/` directory, run:

```bash
mkdir -p models && cd models
wget https://huggingface.co/BrinqAI/functiongemma-270m-physical-ai/resolve/main/functiongemma-physical-ai-Q4_K_M.gguf
```

Source: https://huggingface.co/BrinqAI/functiongemma-270m-physical-ai
Base model: `google/functiongemma-270m-it`
Training data: 367 train / 100 eval, 13-tool physical-AI schema
