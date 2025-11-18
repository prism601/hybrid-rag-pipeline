.PHONY: help install test run docker-build docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make run          - Run API server"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Start Docker containers"
	@echo "  make docker-down  - Stop Docker containers"
	@echo "  make clean        - Clean generated files"

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

test:
	pytest tests/ -v --cov=src --cov-report=html

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -rf chroma_db/ logs/
