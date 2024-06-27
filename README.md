# Diffusion Models Encode the Intrinsic Dimension of Data Manifolds
![ICML 2024](https://img.shields.io/badge/ICML-2024-blue.svg)

This repo is a clean and simplified reimplementation of the official PyTorch codebase for the paper [Diffusion Models Encode the Intrinsic Dimension of Data Manifolds](https://arxiv.org/abs/2212.12611).

by Jan Stanczuk*, Georgios Batzolis*, Teo Deveney, and Carola-Bibiane Schönlieb

You can find the paper on [arXiv](https://arxiv.org/abs/2212.12611) and more details on the project's [website](https://gbatzolis.github.io/ID-diff/).

--------------------

## How to run the code

### Dependencies

Run the following to install a subset of necessary python packages for our code
```sh
pip install -r requirements.txt
```

### Usage
To train a diffusion model, use `train.py`. To extract the intrinsic dimension, use `eval.py`.

### Example
For a complete description of how to train the model and extract the intrinsic dimension from the trained diffusion model, refer to `demo.ipynb`.

## References

If you find the code useful for your research, please consider citing
```bib
@article{stanczuk2022your,
  title={Your diffusion model secretly knows the dimension of the data manifold},
  author={Stanczuk, Jan and Batzolis, Georgios and Deveney, Teo and Sch{\"o}nlieb, Carola-Bibiane},
  journal={arXiv preprint arXiv:2212.12611},
  year={2022}
}
```
