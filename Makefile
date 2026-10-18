.PHONY: review quality notebooks
review:
	python3 -S review/run.py --output /tmp/lava-review --with-node
	python3 -S review/run.py --output /tmp/lava-review --with-node --check
quality:
	bash scripts/quality_gate.sh
notebooks:
	uv run --frozen python scripts/execute_notebooks.py --publish
