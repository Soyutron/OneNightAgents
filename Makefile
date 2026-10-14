.PHONY: sync test play

sync:
	uv sync

test:
	uv run pytest -q

play:
	uv run onenightagents play --reveal
