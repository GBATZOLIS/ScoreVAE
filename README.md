Variational Diffusion Auto-encoder: Latent Space Extraction from Pre-trained Diffusion Models

This repo is an implementation of the paper [Variational Diffusion Auto-encoder: Latent Space Extraction from Pre-trained Diffusion Models](https://arxiv.org/abs/2304.12141).

by Georgios Batzolis*, Jan Stanczuk*, Teo Deveney, and Carola-Bibiane Schönlieb

--------------------

## How to run the code

### Dependencies

Run the following to create the conda environment and install necessary packages:
```sh
conda env create -f environment.yml
conda activate id-diff
```

### Usage
To train a diffusion model, use `train.py`. To evaluate the model, use `eval.py`.

### Example
For a complete description of how to train the model and use it for compression/manipulation/editing, refer to `demo.ipynb`.

## References

If you find the code useful for your research, please consider citing
```bib
@article{batzolis2023variational,
  title={Variational Diffusion Auto-encoder: Latent Space Extraction from Pre-trained Diffusion Models},
  author={Batzolis, Georgios and Stanczuk, Jan and Sch{\"o}nlieb, Carola-Bibiane},
  journal={arXiv preprint arXiv:2304.12141},
  year={2023}
}
```
