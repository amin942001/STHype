# STHype

## Install

```zsh
pip install .
```

## Development

### Environment

#### Set up

conda:

```zsh
conda env create -f environment.yml
```

`pip` virtual environment:

```zsh
python3 -m venv STHypeEnv
source STHypeEnv/bin/activate
pip install -r requirements.txt
```

#### Update

`environment.yml` update:

```zsh
conda env export > environment.yml
```

`requirements.txt`update:

```zsh
pip list --format=freeze > requirements.txt
```

### Tests

Run all tests with :

```zsh
python tests # Add -v for more details
```

Run a specific test with (e.g _test_spatial_graph.py_):

```zsh
python tests/test_spatial_graph.py # Add -v for more details
```

### Formatting

For linting use [Flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8) and [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint).

For auto formatting use [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) and [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort).

For docstring use [autoDocstring - Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) or "numpy" format.

For spelling use [code spell checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker).
