#!/bin/bash
# Clean all generated artifacts
# Run from repo root: bash scripts/clean_artifacts.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Cleaning artifacts..."

if [ -d "$REPO_ROOT/artifacts" ]; then
    rm -rf "$REPO_ROOT/artifacts/logs"/*
    rm -rf "$REPO_ROOT/artifacts/outputs"/*
    rm -rf "$REPO_ROOT/artifacts/evaluation_results"/*
    rm -rf "$REPO_ROOT/artifacts/benchmarks"/*
    rm -rf "$REPO_ROOT/artifacts/checkpoints"/*
    echo "  artifacts/ cleaned"
else
    echo "  artifacts/ does not exist, nothing to clean"
fi

# Also clean Python build artifacts
find "$REPO_ROOT" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find "$REPO_ROOT" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find "$REPO_ROOT" -name "*.pyc" -delete 2>/dev/null || true
echo "  Python cache cleaned"

# Clean chamfer build artifacts
if [ -d "$REPO_ROOT/external/chamfer/build" ]; then
    rm -rf "$REPO_ROOT/external/chamfer/build"
    echo "  chamfer build/ cleaned"
fi
if [ -d "$REPO_ROOT/external/chamfer/dist" ]; then
    rm -rf "$REPO_ROOT/external/chamfer/dist"
    echo "  chamfer dist/ cleaned"
fi
find "$REPO_ROOT/external/chamfer" -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Done."
