# Convenience wrappers only. Every target is a thin alias for a `python -m
# research_agent ...` command, and the README quickstart uses the raw commands --
# GNU make is not installed on Windows by default and the 30-point functionality
# block must never depend on it.
#
# This Makefile contains no version-control commands and never will.

PY ?= python

.DEFAULT_GOAL := help

.PHONY: help setup doctor fetch-corpus ingest index ask chat eval eval-retrieval test budget

help:  ## List available targets
	@echo "cited-research-agent"
	@echo ""
	@echo "  setup           install pinned dependencies into the active venv"
	@echo "  doctor          probe GPU, Ollama, SQLite FTS5, providers, ladders"
	@echo "  fetch-corpus    download the manifest papers from arXiv"
	@echo "  ingest          extract text and page metadata from data/sources"
	@echo "  index           build bge-m3 embeddings and the FTS5 index"
	@echo "  ask             single-turn question -> cited answer"
	@echo "  chat            multi-turn REPL"
	@echo "  eval            full evaluation harness"
	@echo "  eval-retrieval  LLM-free retrieval ablation (zero quota)"
	@echo "  test            pytest, no network"
	@echo "  budget          remaining RPM/RPD per model per key, from the ledger"

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

doctor:
	$(PY) -m research_agent doctor

fetch-corpus:
	@echo "not yet implemented (Step 2)"

ingest:
	@echo "not yet implemented (Step 3)"

index:
	@echo "not yet implemented (Step 4)"

ask:
	@echo "not yet implemented (Step 7)"

chat:
	@echo "not yet implemented (Step 9)"

eval:
	@echo "not yet implemented (Step 12)"

eval-retrieval:
	@echo "not yet implemented (Step 6)"

test:
	$(PY) -m pytest

budget:
	$(PY) -m research_agent budget
