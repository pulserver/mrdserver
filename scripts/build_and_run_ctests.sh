#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MRDSERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$MRDSERVER_DIR/.." && pwd)"
SOURCE_DIR="$MRDSERVER_DIR/tests/cpp"
BUILD_DIR="$REPO_ROOT/build_mrdserver_tests"

echo "Configuring mrdserver C++ tests..."
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug

echo "Building mrdserver C++ tests..."
cmake --build "$BUILD_DIR"

echo "Running mrdserver C++ tests with CTest..."
ctest --test-dir "$BUILD_DIR" --output-on-failure
