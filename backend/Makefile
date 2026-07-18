CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -pedantic

OUT_DIR := build
MAZE_TRAINER := $(OUT_DIR)/maze_trainer
BTC_TRAINER := $(OUT_DIR)/btc_trainer
MAZE_MODEL_JS := ../examples/advanced-maze-model.js
MAZE_MODEL_JSON := ../examples/advanced-maze-model.json
BTC_MODEL_JS := ../examples/btc-trading-model.js
BTC_MODEL_JSON := ../examples/btc-trading-model.json
BTC_CSV := ../dataset/btc_4h.csv

.PHONY: all train train-maze train-btc clean

all: $(MAZE_TRAINER) $(BTC_TRAINER)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(MAZE_TRAINER): maze_trainer.c | $(OUT_DIR)
	$(CC) $(CFLAGS) -o $(MAZE_TRAINER) maze_trainer.c

$(BTC_TRAINER): btc_trainer.c | $(OUT_DIR)
	$(CC) $(CFLAGS) -o $(BTC_TRAINER) btc_trainer.c

train: train-maze train-btc

train-maze: $(MAZE_TRAINER)
	$(MAZE_TRAINER) --js $(MAZE_MODEL_JS) --json $(MAZE_MODEL_JSON)

train-btc: $(BTC_TRAINER)
	$(BTC_TRAINER) --csv $(BTC_CSV) --js $(BTC_MODEL_JS) --json $(BTC_MODEL_JSON)

clean:
	rm -rf $(OUT_DIR)
