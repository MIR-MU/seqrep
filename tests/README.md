# Tests

For testing, you need to install `pytest` and `pytest-cov` packages.


## Testing
To run all tests:

```bash
pytest -v tests/
```

To run a specific test:

```bash
pytest -v ./tests/test_specific_file.py
```

## Test Coverage
To get a test coverage:
```bash
pytest --cov=seqrep/ tests/ 
```
