# CalibQaunt

Code for the paper "[1-Bit KV Cache Quantization for Multimodal LLMs](https://arxiv.org/pdf/2502.14882v1)"

Authors: Insu Han, Zeliang Zhang, Zhiyuan Wang, Yifan Zhu, Susan Liang, Jiani Liu, Haiting Lin, Mingjie Zhao, Chenliang Xu, Kun Wan, Wentian Zhao


This repository provides a guide for setting up and running InternVL with KVcacheQuant for efficient inference.

# Installation

1. Install required packages (e.g., InternVL,  Triton):
```python
pip install internvl triton==3.2.0
```
2. Download the InternVL2.5-26B/8B model from Hugging Face and place it in the checkpoints folder:

```bash
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/OpenGVLab/InternVL2_5-26B
cd ..
```

## Modify Parameters

1. Change batch size
2. Set bit number
3. Run Inference
```python
python infer.py
```

## Notes

- Ensure all dependencies are installed before running the script.
- Modify parameters accordingly for optimal performance based on your hardware.
- If you encounter issues, refer to the official documentation or repository.

