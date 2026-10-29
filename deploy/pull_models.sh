#!/usr/bin/env bash
export OLLAMA_MODELS=/workspace/.ollama/models
set -e
/usr/local/bin/ollama pull qwen2.5:3b
/usr/local/bin/ollama pull bge-m3
echo ALL_PULLS_DONE
