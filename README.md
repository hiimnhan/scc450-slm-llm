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

# Convert PDF files into multiple formats

_Basic Usage_

```bash
uv run convert_docs.py --input <pdf_root_directory> --output <output_directory>
```

This will convert all PDFs under <pdf_root_directory> into both Markdown (.md) and JSON (.json) formats by default.

_Example_

```bash
uv run convert_docs.py --input ./TraffordCouncilPlanningApplicationsWA14/TraffordCouncil/ --output ./TraffordCouncilPlanningApplicationsWA14/converted
```

_Convert into multiple formats_
You can specify multiple output formats using the `--formats` or `-f` flag. List of output format supported can be found here: [output formats](https://docling-project.github.io/docling/usage/supported_formats/)

```bash
uv run convert_docs.py --input ./TraffordCouncilPlanningApplicationsWA14/TraffordCouncil/ --output ./TraffordCouncilPlanningApplicationsWA14/converted -f json md txt html
```

_Show help_

```bash
uv run convert_docs.py --help
```

```bash
usage: convert_docs.py [-h] --input_folder INPUT_FOLDER [--formats {md,json,txt,doctags} [{md,json,txt,doctags} ...]] --output_folder OUTPUT_FOLDER

Convert PDF documents to other formats.

options:
  -h, --help            show this help message and exit
  --input_folder INPUT_FOLDER, -i INPUT_FOLDER
                        Path to the input folder containing PDF documents.
  --formats {md,json,txt,doctags} [{md,json,txt,doctags} ...], -f {md,json,txt,doctags} [{md,json,txt,doctags} ...]
  --output_folder OUTPUT_FOLDER, -o OUTPUT_FOLDER
                        Path to the output folder for converted documents.
```
