# STHype

## Install

```
pip install .
```

## Development

### Environment

conda:

```
conda env create -f environment.yml
```

`pip` virtual environment:

```
python3 -m venv STHypeEnv
source STHypeEnv/bin/activate
pip install -r requirements.txt
```

### Tests

Run all tests with :

```
python tests
```

Run a specific test with (e.g _tests_spatial_graph.py_):

```
python tests/tests_spatial_graph.py
```
