.PHONY: setup init data run ui clean

setup:
	pip install -r requirements.txt

init:
	git init
	dvc init
	git add .
	git commit -m "chore: project scaffold"

data:
	python src/generate_data.py
	dvc add data/raw/clickstream.csv
	git add data/raw/clickstream.csv.dvc .gitignore
	git commit -m "data: add raw clickstream v1 (5000 rows)"

run:
	dvc repro
	dvc push
	git add dvc.lock params.yaml metrics/
	git commit -m "run: pipeline complete"

ui:
	mlflow ui

diff:
	dvc metrics diff HEAD~1

clean:
	rm -rf data/processed/ models/ metrics/ mlruns/
