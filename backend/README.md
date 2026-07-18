# C Backend Trainer

This folder contains small native trainers that export action-value weights `tinyfin-js` can load as one-layer dense Q-networks in the browser.

Build and export all pretrained models:

```sh
make train
```

Build only the advanced maze model:

```sh
make train-maze
```

Build only the BTC trading model:

```sh
make train-btc
```

You can also run the compiled trainer directly:

```sh
cd build
./maze_trainer
```

Outputs:

- `../examples/advanced-maze-model.js`: browser-ready model loaded by `advanced-maze.html`.
- `../examples/advanced-maze-model.json`: plain JSON copy for tools or inspection.
- `../examples/btc-trading-model.js`: browser-ready model loaded by `btc-trading.html`.
- `../examples/btc-trading-model.json`: plain JSON copy for tools or inspection.

The exported weights are shaped as `[action][state]`. In the browser, a `DQNAgent` with `hiddenSizes: []` uses those rows directly as the dense output layer, so `agent.policy(oneHotState)` returns the trained Q-values.

This is a backend trainer in the narrow, practical sense: native training/export for browser inference. It is not yet a general C implementation of the full neural DQN replay/backprop pipeline.

The BTC trainer reads `../dataset/btc_4h.csv`, keeps the latest 720 rows, and computes a future-aware dynamic-programming policy over `(time, position)` states. That makes it useful as an offline oracle/baseline for the browser demo, not as a realistic live trading strategy.
