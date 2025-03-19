# CalibQuant: 1-Bit KV Cache Quantization via Calibration for Multimodal LLMs 

Code for the paper "[1-Bit KV Cache Quantization for Multimodal LLMs](https://arxiv.org/pdf/2502.14882v1)"

Authors: Insu Han, Zeliang Zhang, Zhiyuan Wang, Yifan Zhu, Susan Liang, Jiani Liu, Haiting Lin, Mingjie Zhao, Chenliang Xu, Kun Wan, Wentian Zhao


This repository provides a guide for setting up and running InternVL with KVcacheQuant for efficient inference.

## Installation

1. Install required packages (e.g., InternVL,  Triton):
```python
pip install internvl triton==3.2.0
```

2. Download the InternVL2.5-26B/8B model from HuggingFace:


## Modify Parameters

1. Change batch size (line 104 in ``infer.py``)
2. Set bit number (line 13 in ``calibquant.py``)
3. Run Inference
```python
python infer.py
```

## Notes

- Ensure all dependencies are installed before running the script.
- Modify parameters accordingly for optimal performance based on your hardware.
- If you encounter issues, refer to the official documentation or repository.

## Citation
```bib
@article{,
  title={From 16-Bit to 1-Bit: Visual KV Cache Quantization for Memory-Efficient Multimodal Large Language Models},
  author={Zhang, Zeliang and Zhu, Yifan and Liang, Susan and Wang, Zhiyuan and Liu, Jiani and Lin, Haiting and Zhao, Mingjie and Xu, Chenliang and Wan, Kun and Zhao, Wentian},
  journal={arXiv preprint arXiv:2502.14882},
  year={2025}
}
```
