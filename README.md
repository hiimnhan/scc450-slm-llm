# Installation

1. Create new environment

```bash
python -m venv .venv
```

2. Activate environment

```bash
source .venv/bin/activate
```

2. Install required packages

```bash
pip install -r requirements.txt
```

3. (Optional) install new packages

- Add your package into `requirements.in` file
- Run [pip-compile](https://pypi.org/project/pip-tools/) command to add that package and its dependencies into `requirements.txt` (see link if you haven't had it installed)

```bash
pip-compile requirements.in -o requirements.txt
```

- Re-run step 2
