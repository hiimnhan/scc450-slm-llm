# Installation

1. Install `uv`. [uv](https://docs.astral.sh/uv/) is a fast Python package and environment manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
```

2. Create and activate environment

- Create a new environment

```bash
uv venv
```

- Activate (Linux/macOS)

```bash
source .venv/bin/activate
```

- Activate (Windows)

```bash
.venv\Scripts\activate
```

3. Install project dependencies

```bash
uv sync
```

To install a package

```bash
uv add package_name
```
