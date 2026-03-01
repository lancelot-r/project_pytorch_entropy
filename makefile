PYTHON=python

DATA_CONFIG=config/config_data.json
TRAIN_CONFIG=config/config_train.json
EVAL_CONFIG=config/config_eval.json

DATA_CONFIG_MULTI=config/config_data_multi.json
TRAIN_CONFIG_MULTI=config/config_train_multi.json
EVAL_CONFIG_MULTI=config/config_eval_multi.json

.PHONY: all clean univariate multivariate \
        data train eval \
        data_multi train_multi eval_multi

all: univariate

univariate: data train eval

multivariate: data_multi train_multi eval_multi

data:
	$(PYTHON) code/metadata.py --config $(DATA_CONFIG)

train:
	$(PYTHON) code/metastat_main.py --config $(TRAIN_CONFIG)

eval:
	$(PYTHON) code/evaluation.py --config $(EVAL_CONFIG)

data_multi:
	$(PYTHON) code/multi_metadata.py --config $(DATA_CONFIG_MULTI)

train_multi:
	$(PYTHON) code/multi_metastat_main.py --config $(TRAIN_CONFIG_MULTI)

eval_multi:
	$(PYTHON) code/multi_evaluation.py --config $(EVAL_CONFIG_MULTI)

clean:
	rm -rf training/*
	rm -rf evaluation/*