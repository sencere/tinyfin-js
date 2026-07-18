# C Backend Trainer

This folder contains a small native trainer for the advanced maze demo. It runs value iteration over the maze MDP and exports action-value weights that `tinyfin-js` can load as a one-layer dense Q-network in the browser.

Build and export the pretrained model:

```sh
make train
```

You can also run the compiled trainer directly:

```sh
cd build
./maze_trainer
```

Outputs:

- `../examples/advanced-maze-model.js`: browser-ready model loaded by `advanced-maze.html`.
- `../examples/advanced-maze-model.json`: plain JSON copy for tools or inspection.

The exported weights are shaped as `[action][state]`. In the browser, a `DQNAgent` with `hiddenSizes: []` uses those rows directly as the dense output layer, so `agent.policy(oneHotState)` returns the trained Q-values.

This is a backend trainer in the narrow, practical sense: native training/export for browser inference. It is not yet a general C implementation of the full neural DQN replay/backprop pipeline.
