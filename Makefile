test:
	uv run pytest tests
lint:
	uv run ruff check .
	uv run ruff format .
	uv run pyright .
publish:
	uv build
	uv publish
