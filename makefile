PYTHON = python
CONFIG = config.json
DATA_GENERATION = metadata.py
DATA_LOAD = metastat_dataloader.py
MAIN = metastat_main.py
MODEL = metastat_model.py
MODEL_CONFIG = model_params.json
MODEL_TEST = test_script.py
data:
	$(PYTHON) $(DATA_GENERATION) --config $(CONFIG)

model:
	$(PYTHON) $(MODEL)

train:
	$(PYTHON) $(MAIN) --config $(MODEL_CONFIG)

test:
	$(PYTHON) $(MODEL_TEST)