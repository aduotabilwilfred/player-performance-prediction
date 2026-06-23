all: data train test

setup:
	uv venv ml-env
	uv pip compile requirements.in -o requirements.txt
	uv pip sync requirements.txt
	uv pip install -e .

data:
	dvc repro
train:
	python3 -m player_prediction.models.train.py
test:
	pytest tests/
clean:
	rm -rf __pycache__ .pytest_cache

.PHONY: all setup data train test clean
