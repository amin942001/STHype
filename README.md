# STHype

## Install

```
pip install .
```

## Development

### Environment

#### Set up

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

#### Update

`environment.yml` update:

```
conda env export > environment.yml
```

`requirements.txt`update:

```
pip list --format=freeze > requirements.txt
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
