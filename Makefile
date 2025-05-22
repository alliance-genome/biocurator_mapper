.PHONY: test test-cov test-fast clean install lint

# Virtual environment activation  
SHELL := /bin/bash
VENV_ACTIVATE = source /home/ctabone/.virtualenvs/flybase/bin/activate &&

# Install dependencies
install:
	$(VENV_ACTIVATE) pip install -r requirements.txt

# Run all tests
test:
	$(VENV_ACTIVATE) pytest tests/test_config.py tests/test_models.py tests/test_go_data_parsing.py tests/test_yaml_config_integration.py

# Run tests with coverage
test-cov:
	$(VENV_ACTIVATE) pytest tests/test_config.py tests/test_models.py tests/test_go_data_parsing.py tests/test_yaml_config_integration.py --cov=app --cov-report=html --cov-report=term

# Run only fast tests (exclude slow markers)
test-fast:
	$(VENV_ACTIVATE) pytest tests/test_config.py tests/test_models.py tests/test_go_data_parsing.py tests/test_yaml_config_integration.py -m "not slow"

# Run specific test file
test-config:
	$(VENV_ACTIVATE) pytest tests/test_config.py -v

test-main:
	$(VENV_ACTIVATE) pytest tests/test_main.py -v

test-models:
	$(VENV_ACTIVATE) pytest tests/test_models.py -v

test-go-parsing:
	$(VENV_ACTIVATE) pytest tests/test_go_data_parsing.py -v

test-yaml-integration:
	$(VENV_ACTIVATE) pytest tests/test_yaml_config_integration.py -v

# Clean up generated files
clean:
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Lint code (if you want to add linting later)
lint:
	@echo "Add your preferred linter here (flake8, black, etc.)"