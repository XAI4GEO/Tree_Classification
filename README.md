# Tree Classification workflow

This repository contains the workflow for the research work "Explainable few-shot learning workflow for detecting invasive and exotic tree species". 

## Repository structure

- `notebooks`: Jupyter notebooks to replicate the research work.
- [`packages`](./packages): Python tools that are used across the project. 

## Setup

The Python environment for the workflow can be set up using the `environment.yml` file. We recommand using `mamba` to create the environment. The environment can be created using the following command:

```bash
mamba env create -f environment.yml
```

## Data and tools used to train the Siamese model

The [Netflora workflow](https://github.com/NetFlora/Netflora) is used to generate part of the training dataset.

The [ReforesTree](https://github.com/gyrrei/ReforesTree) dataset is used as part of the training dataset.
