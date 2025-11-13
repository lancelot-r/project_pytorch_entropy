PYTHON = python
CONFIG = config.json
DATA_GENERATION = metadata.py
DATA_LOAD = metastat_dataloader.py
MAIN = metastat_main.py
MODEL = metastat_model.py
MODEL_CONFIG = model_params.json
train_data:
	$(PYTHON) $(DATA_GENERATION) --config $(CONFIG)

load_data:
	$(PYTHON) $(DATA_LOAD)

model:
	$(PYTHON) $(MODEL)

train_model:
	$(PYTHON) $(MAIN) --config $(MODEL_CONFIG)