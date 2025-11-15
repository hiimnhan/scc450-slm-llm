= Environment Setup
This project used _Conda_ for environment management.

1. Create the conda environment

```bash
conda env create -f environment.yml
```

2. Activate the environment

```bash
conda activate scc450-slm-llm
```

3. (Optional) Update the environment after changes
   If you edit `environment.yml`, update the existing env

```bash
conda env update -f environment.yml --prune
```
