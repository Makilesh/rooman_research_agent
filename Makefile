# Convenience wrappers only. Every target is a thin alias for a `python -m
# research_agent ...` command, and the README quickstart uses the raw commands --
# GNU make is not installed on Windows by default and the 30-point functionality
# block must never depend on it.
#
# This Makefile contains no version-control commands and never will.

PY ?= python

# `make ask` needs a question. Override it:
#   make ask Q="What rank does LoRA use in its GPT-3 175B experiments?"
Q ?= What rank does LoRA use in its GPT-3 175B experiments, and which weight matrices does it adapt?

.DEFAULT_GOAL := help

.PHONY: help setup doctor fetch-corpus ingest index ask chat eval eval-retrieval \
        answer-all chat-eval test budget

help:  ## List available targets
	@echo "cited-research-agent"
	@echo ""
	@echo "  setup           install pinned dependencies into the active venv"
	@echo "  doctor          probe GPU, Ollama, SQLite FTS5, providers, ladders"
	@echo "  fetch-corpus    download the manifest papers from arXiv"
	@echo "  ingest          extract text and page metadata from data/sources"
	@echo "  index           build bge-m3 embeddings and the FTS5 index"
	@echo "  ask             single-turn question -> cited answer   (Q=\"...\")"
	@echo "  chat            multi-turn REPL"
	@echo "  answer-all      regenerate outputs/answers/ from data/questions.yaml"
	@echo "  chat-eval       run the four conversation scenarios"
	@echo "  eval            full evaluation harness"
	@echo "  eval-retrieval  LLM-free retrieval ablation (zero quota)"
	@echo "  test            pytest, no network"
	@echo "  budget          remaining RPM/RPD per model per key, from the ledger"

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

doctor:
	$(PY) -m research_agent doctor --gpu

fetch-corpus:
	$(PY) -m research_agent fetch

ingest:
	$(PY) -m research_agent ingest

index:
	$(PY) -m research_agent index

ask:
	$(PY) -m research_agent ask "$(Q)"

chat:
	$(PY) -m research_agent chat

answer-all:
	$(PY) -m research_agent answer-all

chat-eval:
	$(PY) -m research_agent chat-eval

eval:
	$(PY) -m research_agent eval

eval-retrieval:
	$(PY) -m research_agent eval-retrieval

test:
	$(PY) -m pytest

budget:
	$(PY) -m research_agent budget
