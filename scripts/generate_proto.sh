#!/bin/bash
# Script to generate Python gRPC code from proto files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_DIR="$(dirname "$CLIENT_DIR")/server"
PROTO_FILE="$SERVER_DIR/api/serverside.proto"
OUTPUT_DIR="$CLIENT_DIR/grpc_client"

echo "Generating Python gRPC code from proto..."
echo "Proto file: $PROTO_FILE"
echo "Output dir: $OUTPUT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Generate Python code
python -m grpc_tools.protoc \
    -I"$SERVER_DIR/api" \
    --python_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    "$PROTO_FILE"

echo "✓ Generated serverside_pb2.py and serverside_pb2_grpc.py"
echo "Done!"
