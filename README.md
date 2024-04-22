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
python tests # Add -v for more details
```

Run a specific test with (e.g _test_spatial_graph.py_):

```
python tests/test_spatial_graph.py
```
