SHELL := /bin/bash

toml-sort:
	uv run toml-sort -i pyproject.toml --no-sort-tables --sort-table-keys

lock-requirements:
	uv export --format requirements-txt > requirements.txt
