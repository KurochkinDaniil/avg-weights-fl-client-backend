.PHONY: install proto run-api federated-cycle clean help

help:
	@echo "Available commands:"
	@echo "  make install          - Install Python dependencies"
	@echo "  make proto            - Generate gRPC Python code from proto files"
	@echo "  make run-api          - Run FastAPI server"
	@echo "  make federated-cycle  - Run federated learning cycle"
	@echo "  make clean            - Clean generated files and cache"

install:
	pip install -r requirements.txt

proto:
	bash scripts/generate_proto.sh

run-api:
	python scripts/run_api.py

federated-cycle:
	python scripts/federated_cycle.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -f grpc_client/*_pb2.py grpc_client/*_pb2_grpc.py
