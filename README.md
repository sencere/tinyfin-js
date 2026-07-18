# tinyfin-js

`tinyfin-js` is a small browser-first reinforcement learning library for discrete-action Deep Q-Network experiments. It ships as a single JavaScript file, has no runtime dependencies, and exposes a global `tinyfin` object when loaded with a script tag.

The project is intentionally compact and inspectable. It is meant for toy environments, teaching demos, small browser games, and pretrained policy visualizations rather than large-scale RL workloads.

## Features

- Browser-native JavaScript, no Node runtime required.
- Dense neural Q-network with ReLU hidden layers.
- Experience replay.
- Epsilon-greedy exploration.
- Target network updates.
- Optional Double DQN target calculation.
- Adam or SGD updates.
- Temporal state/action window support.
- JSON save/load for agents.
- Seeded random helper for reproducible demos.
- Native C backend trainer for generating a pretrained browser model.

## Files

- `tinyfin.js`: the browser library.
- `examples/gridworld.html`: trains a small DQN live in the browser.
- `examples/pretrained-gridworld.html`: loads an already solved policy.
- `examples/advanced-maze.html`: advanced pretrained maze visualization.
- `examples/advanced-maze-model.js`: generated browser model for the advanced demo.
- `examples/advanced-maze-model.json`: generated model data for tools or inspection.
- `backend/maze_trainer.c`: native C trainer/exporter for the advanced maze model.

## Quick Start

Use the library directly from HTML:

```html
<script src="./tinyfin.js"></script>
<script>
  const agent = new tinyfin.DQNAgent({
    stateSize: 4,
    actionSize: 2,
    hiddenSizes: [32, 32],
    learningRate: 0.001,
    gamma: 0.99,
    trainStart: 200
  });

  let state = env.reset();

  function tick() {
    const action = agent.act(state);
    const result = env.step(action);

    agent.learnFromReward(result.reward, result.state, result.done);
    agent.observe(result.reward);

    state = result.done ? env.reset() : result.state;
  }
</script>
```

No bundler or package manager is required. Opening the example HTML files directly in a browser is enough.

## Basic Loop

A DQN agent needs numeric state vectors, discrete action indices, scalar rewards, and terminal flags:

```js
const action = agent.act(state);
const result = environment.step(action);

agent.learnFromReward(result.reward, result.state, result.done);

if (result.done) {
  agent.resetEpisode();
  state = environment.reset();
} else {
  state = result.state;
}
```

For offline or custom replay data, use `step()` directly:

```js
agent.step(state, action, reward, nextState, done);
```

For inference without exploration:

```js
const decision = agent.policy(state);
console.log(decision.action, decision.values);
```

## API

### `new tinyfin.DQNAgent(options)`

Required options:

- `stateSize`: number of numeric inputs in each observation.
- `actionSize`: number of discrete actions.

Useful options:

- `hiddenSizes`: hidden layer sizes, default `[32]`.
- `gamma`: future reward discount, default `0.99`.
- `epsilon`: exploration probability, default `1`.
- `epsilonMin`: lowest exploration probability, default `0.05`.
- `epsilonDecay`: multiplicative decay after training updates, default `0.995`.
- `replaySize`: maximum replay memory length, default `10000`.
- `batchSize`: replay samples per training update, default `32`.
- `trainStart`: replay size required before training begins, default `100`.
- `learningRate`: optimizer step size, default `0.001`.
- `targetUpdateInterval`: training updates between target copies, default `100`.
- `doubleDQN`: use Double DQN targets, default `true`.
- `temporalWindow`: append recent state/action history to inputs, default `0`.
- `randomActionDistribution`: optional weighted random action distribution.
- `optimizer`: `"adam"` or `"sgd"`, default `"adam"`.
- `gradientClip`: scalar gradient clipping bound, default `10`.
- `l2`: weight decay, default `0`.
- `rng`: custom random function.

Common methods:

- `act(state, learn = true)`: returns an action index. Pass `false` to use greedy evaluation epsilon.
- `learnFromReward(reward, nextState, done)`: records the transition after the last `act()` call and trains when replay is ready.
- `step(state, action, reward, nextState, done)`: records an explicit transition and trains when replay is ready.
- `learn()`: samples replay and applies one training pass when enough data exists.
- `policy(state)`: returns `{ action, values }` without random exploration.
- `observe(reward)`: records reward statistics.
- `stats()`: returns counters, epsilon, replay size, reward average, and loss average.
- `updateTarget()`: copies online network weights to the target network.
- `resetEpisode()`: clears temporal state/action history.
- `toJSON()` / `fromJSON(json)`: save and restore an agent.

### Utilities

- `tinyfin.DenseNetwork`: small dense network used by `DQNAgent`.
- `tinyfin.ReplayBuffer`: fixed-size random replay buffer.
- `tinyfin.RollingAverage`: simple rolling average helper.
- `tinyfin.argmax(values)`: returns the index of the largest value.
- `tinyfin.createSeededRandom(seed)`: deterministic random number generator.

## Examples

### Live Training

Open `examples/gridworld.html`.

This demo trains in the browser. It shows the normal DQN workflow: choose an action, step the environment, add a transition to replay, and train from sampled experiences.

### Pretrained Gridworld

Open `examples/pretrained-gridworld.html`.

This demo starts with a solved model. The policy is represented as dense Q-network weights and is run through `agent.policy(state)`.

### Advanced Maze

Open `examples/advanced-maze.html`.

This demo uses a model generated by the C backend trainer. It includes walls, hazard states, random starts, a path trace, optional policy arrows, an optional value heatmap, and live action-value bars.

## C Backend Trainer

The backend trainer generates the advanced maze model natively:

```sh
cd backend
make train
```

This builds `backend/build/maze_trainer` and regenerates:

- `examples/advanced-maze-model.js`
- `examples/advanced-maze-model.json`

The trainer currently solves a known maze MDP with value iteration and exports action-value weights. The generated weights have shape `[action][state]`. In the browser, `advanced-maze.html` creates a `DQNAgent` with `hiddenSizes: []`, installs those rows as the output layer weights, and uses one-hot states for inference.

The C trainer is useful for fast native model generation and browser inference. It is not yet a general neural DQN replay/backprop trainer.

## Model Save And Load

Browser-trained agents can be serialized:

```js
const saved = agent.toJSON();
localStorage.setItem("agent", JSON.stringify(saved));
```

And restored:

```js
const restored = new tinyfin.DQNAgent({
  stateSize: 4,
  actionSize: 2
});

restored.fromJSON(JSON.parse(localStorage.getItem("agent")));
```

For pretrained one-layer policies, the model data can also be exported as plain action-value weights and installed directly into `agent.online.layers[0].weights`.

## Design Notes

`tinyfin-js` takes inspiration from Andrej Karpathy's ConvNetJS Deep Q demo, but keeps the code self-contained and browser-native. The goal is a useful small library with readable internals, not a full machine learning framework.

Use it when you want:

- a reinforcement learning demo that runs from a plain HTML file
- an inspectable DQN implementation
- toy-scale browser training
- pretrained policy playback and visualization
- a small bridge between native training/export and browser inference

## Limits

- Discrete actions only.
- Dense networks only.
- CPU JavaScript only in the browser.
- No convolutional layers, recurrent layers, WebGPU, or tensor backend.
- Best suited to toy environments and educational demos.

For large neural networks, image observations, or serious RL workloads, use a larger stack such as TensorFlow.js, WebGPU-backed inference, or a native training framework.

## License

MIT. See `LICENSE`.
