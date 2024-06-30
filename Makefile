# Makefile

# Variables
PROTO_GEN_DIR := ./pb
PROTO_DIR := ./proto
PROTO_FILE := $(PROTO_DIR)/*.proto
VENV_DIR := ./venv

# Python specific variables
PYTHON := python3
PIP := $(VENV_DIR)/bin/pip
PYTHON_VENV := $(VENV_DIR)/bin/python

.PHONY: all clean proto venv

all: clean venv proto

# Clean generated files and virtual environment
clean:
	rm -f $(PROTO_GEN_DIR)/*_pb2*.py
	rm -rf $(VENV_DIR)

# Set up Python virtual environment
venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating Python virtual environment..."; \
		virtualenv $(VENV_DIR); \
		echo "Activating virtual environment and installing dependencies..."; \
		. $(VENV_DIR)/bin/activate && \
		$(PIP) install --upgrade pip && \
		$(PIP) install grpcio grpcio-tools && \
		$(PIP) install -r requirements.txt; \
	else \
		echo "Python virtual environment already exists."; \
	fi

# Generate Protocol Buffer code
proto: venv $(PROTO_FILE)
	@echo "Generating Python gRPC code..."
	$(PYTHON_VENV) -m grpc_tools.protoc -I$(PROTO_DIR) --python_out=$(PROTO_GEN_DIR) --grpc_python_out=$(PROTO_GEN_DIR) $(PROTO_FILE)
