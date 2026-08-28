.PHONY: setup gate test lint frame criteo panel mmm uplift clv experiments recovery \
        triangulate notebooks-src notebooks reports card dashboard all clean

setup:
	uv sync --all-extras
	uv run python scripts/install_kernel.py

# Import and exercise the four dependencies that can install cleanly and then
# fail on first use. Records the outcome to metrics/env_gate.json rather than
# letting a missing comparison quietly vanish from the report.
gate:
	uv run python scripts/check_env.py

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

frame:
	uv run python scripts/build_frame.py

# The Criteo CSV is 3.2 GB and is read exactly once, here. Everything downstream
# reads the Parquet this writes.
criteo:
	uv run python scripts/build_criteo.py

panel:
	uv run python scripts/build_panel.py

# The headline media-mix fit, on the panel with the real Olist baseline.
mmm:
	uv run python scripts/build_mmm.py

uplift:
	uv run python scripts/build_uplift.py

clv:
	uv run python scripts/build_clv.py

experiments:
	uv run python scripts/build_experiments.py

# Reconciliation, allocation, and the artifact the dashboard reads.
triangulate:
	uv run python scripts/build_triangulation.py

# 40 full-posterior fits. Resumable: each cell is cached under a hash of its
# configuration, so an interrupted run costs one fit rather than the batch.
recovery:
	uv run python scripts/run_recovery.py

# The notebooks are generated from one place, because eight of them share a
# preamble and a house style and eight hand-maintained copies drift.
notebooks-src:
	uv run python scripts/build_notebooks.py

# Execute, then clear, as two passes. Combining them does not work: nbconvert
# runs ClearOutputPreprocessor before ExecutePreprocessor, so a single invocation
# clears the outputs and then executes straight over them. Committed notebooks
# carry no outputs; the results live in metrics/ and reports/.
notebooks:
	JUPYTER_PATH=$(CURDIR)/.jupyter uv run jupyter nbconvert --to notebook \
		--execute --inplace --ExecutePreprocessor.kernel_name=athar notebooks/*.ipynb
	uv run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

# Every figure in reports/ is read from metrics/. Deleting reports/ and rerunning
# reproduces them byte for byte.
reports:
	uv run python scripts/render_reports.py

card:
	uv run python scripts/render_card.py

dashboard:
	cd dashboard && npm ci && npm run build

all: lint test gate frame criteo panel mmm uplift clv experiments recovery \
     triangulate notebooks-src notebooks reports card dashboard

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ .coverage
