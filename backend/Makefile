CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -pedantic

OUT_DIR := build
TRAINER := $(OUT_DIR)/maze_trainer
MODEL_JS := ../examples/advanced-maze-model.js
MODEL_JSON := ../examples/advanced-maze-model.json

.PHONY: all train clean

all: $(TRAINER)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(TRAINER): maze_trainer.c | $(OUT_DIR)
	$(CC) $(CFLAGS) -o $(TRAINER) maze_trainer.c

train: $(TRAINER)
	$(TRAINER) --js $(MODEL_JS) --json $(MODEL_JSON)

clean:
	rm -rf $(OUT_DIR)
