PYTHON ?= python3

.PHONY: tasks e2e bench report server tests

server:
	$(PYTHON) server_py/app.py --port 3001 --tools tools.jsonl

tasks:
	PYTHONPATH=. $(PYTHON) bench/taskgen.py --tools tools.jsonl --generic-out tests/tasks/generic_tasks.jsonl --fuzzy-out tests/tasks/fuzzy_tasks.jsonl --tiers realistic --seed 42

e2e:
	PYTHONPATH=. $(PYTHON) bench/run_e2e.py --config configs/local_qwen.yaml

bench:
	PYTHONPATH=. $(PYTHON) bench/run_bench.py --config configs/ablation.yaml

report:
	PYTHONPATH=. $(PYTHON) bench/report.py --runs runs --out runs/report

tests:
	PYTHONPATH=. pytest
