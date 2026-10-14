.PHONY: sync test play bench

sync:
	uv sync

test:
	uv run pytest -q

play:
	uv run --env-file .env onenightagents play --reveal --personas configs/personas

bench:
	uv run --env-file .env onenightagents bench --games 5 --personas configs/personas
