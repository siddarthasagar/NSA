main:
	uv venv --clear
	uv sync

add-deps:
	uv add -r requirements.txt

dev:
	uv venv --clear
	uv sync --dev

# format:
# 	uv run ruff format ArcAgent.py ArcAgentFeatureExtractor.py ArcAgentTypes.py tests/test_arc_driver.py
# 	uv run ruff check --fix ArcAgent.py ArcAgentFeatureExtractor.py ArcAgentTypes.py tests/test_arc_driver.py
# # 	uv run ruff format versions
# # 	uv run ruff check --fix versions

# refresh:
# 	rm -rf Milestones
# 	rm -rf visuals
# 	cp -r ~/Downloads/ArcAgi/ArcAgi_Starter_Code_v1.2.2/Milestones .
# 	cp -r ~/Downloads/ArcAgi/ArcAgi_Starter_Code_v1.2.2/visuals .

# refresh-py:
# # 	rm -rv *.py
# 	cp -rf ~/Downloads/ArcAgi/ArcAgi_Starter_Code_v1.2.2/*.py .

.PHONY: train

train:
	uv run python -m small_transformer_based.train $(ARGS)

generate-data:
	uv run python -m auxilaries.generate_transformation $(ARGS)
