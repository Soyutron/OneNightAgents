.PHONY: sync test

sync:
	uv sync

test:
	uv run pytest -q
