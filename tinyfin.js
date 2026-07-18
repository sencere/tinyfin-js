/*
 * tinyfin-js
 * A small browser-first Deep Q-Network library.
 * MIT License
 */
(function (root, factory) {
  if (typeof define === "function" && define.amd) {
    define([], factory);
  } else {
    root.tinyfin = factory();
  }
}(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function assert(condition, message) {
    if (!condition) throw new Error(message);
  }

  function defaults(options, base) {
    var out = {};
    var k;
    options = options || {};
    for (k in base) out[k] = base[k];
    for (k in options) out[k] = options[k];
    return out;
  }

  function zeros(n) {
    var out = new Array(n);
    for (var i = 0; i < n; i++) out[i] = 0;
    return out;
  }

  function copyArray(a) {
    return Array.prototype.slice.call(a, 0);
  }

  function argmax(values) {
    var best = 0;
    var bestValue = values[0];
    for (var i = 1; i < values.length; i++) {
      if (values[i] > bestValue) {
        best = i;
        bestValue = values[i];
      }
    }
    return best;
  }

  function randn(rng) {
    var u = 0;
    var v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function clamp(x, lo, hi) {
    return Math.max(lo, Math.min(hi, x));
  }

  function sampleWeighted(weights, rng) {
    var s = 0;
    var i;
    for (i = 0; i < weights.length; i++) s += weights[i];
    assert(Math.abs(s - 1) < 1e-6, "randomActionDistribution must sum to 1");
    var p = rng();
    var c = 0;
    for (i = 0; i < weights.length; i++) {
      c += weights[i];
      if (p <= c) return i;
    }
    return weights.length - 1;
  }

  function ReplayBuffer(capacity, rng) {
    assert(capacity > 0, "ReplayBuffer capacity must be positive");
    this.capacity = capacity;
    this.rng = rng || Math.random;
    this.items = [];
    this.nextIndex = 0;
  }

  ReplayBuffer.prototype = {
    push: function (transition) {
      if (this.items.length < this.capacity) {
        this.items.push(transition);
      } else {
        this.items[this.nextIndex] = transition;
      }
      this.nextIndex = (this.nextIndex + 1) % this.capacity;
    },

    sample: function (n) {
      var out = new Array(n);
      for (var i = 0; i < n; i++) {
        out[i] = this.items[Math.floor(this.rng() * this.items.length)];
      }
      return out;
    },

    size: function () {
      return this.items.length;
    },

    clear: function () {
      this.items.length = 0;
      this.nextIndex = 0;
    },

    toJSON: function () {
      return {
        capacity: this.capacity,
        items: this.items,
        nextIndex: this.nextIndex
      };
    },

    fromJSON: function (json) {
      this.capacity = json.capacity;
      this.items = json.items || [];
      this.nextIndex = json.nextIndex || 0;
      return this;
    }
  };

  function DenseNetwork(layerSizes, options) {
    options = defaults(options, {
      rng: Math.random,
      activation: "relu",
      outputActivation: "linear",
      learningRate: 0.001,
      gradientClip: 10,
      l2: 0,
      optimizer: "adam",
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8
    });

    assert(layerSizes.length >= 2, "DenseNetwork needs at least input and output layers");
    for (var i = 0; i < layerSizes.length; i++) {
      assert(layerSizes[i] > 0 && Math.floor(layerSizes[i]) === layerSizes[i], "Layer sizes must be positive integers");
    }

    this.layerSizes = layerSizes.slice();
    this.options = options;
    this.layers = [];
    this.t = 0;

    for (i = 1; i < layerSizes.length; i++) {
      var inputSize = layerSizes[i - 1];
      var outputSize = layerSizes[i];
      var scale = Math.sqrt(2 / inputSize);
      var weights = new Array(outputSize);
      var biases = zeros(outputSize);
      var mw = new Array(outputSize);
      var vw = new Array(outputSize);
      var mb = zeros(outputSize);
      var vb = zeros(outputSize);
      for (var o = 0; o < outputSize; o++) {
        weights[o] = new Array(inputSize);
        mw[o] = zeros(inputSize);
        vw[o] = zeros(inputSize);
        for (var j = 0; j < inputSize; j++) {
          weights[o][j] = randn(options.rng) * scale;
        }
      }
      this.layers.push({ weights: weights, biases: biases, mw: mw, vw: vw, mb: mb, vb: vb });
    }
  }

  DenseNetwork.prototype = {
    forward: function (input) {
      assert(input.length === this.layerSizes[0], "Network input length mismatch");
      var activations = [copyArray(input)];
      var preActivations = [];
      var a = activations[0];

      for (var l = 0; l < this.layers.length; l++) {
        var layer = this.layers[l];
        var z = zeros(layer.biases.length);
        var next = zeros(layer.biases.length);
        for (var o = 0; o < layer.biases.length; o++) {
          var sum = layer.biases[o];
          for (var j = 0; j < a.length; j++) sum += layer.weights[o][j] * a[j];
          z[o] = sum;
          next[o] = this.activate(sum, l === this.layers.length - 1);
        }
        preActivations.push(z);
        activations.push(next);
        a = next;
      }

      return { output: activations[activations.length - 1], activations: activations, preActivations: preActivations };
    },

    predict: function (input) {
      return this.forward(input).output;
    },

    trainMSE: function (input, target) {
      var cache = this.forward(input);
      var output = cache.output;
      assert(target.length === output.length, "Target length mismatch");

      var deltas = new Array(this.layers.length);
      var last = this.layers.length - 1;
      deltas[last] = zeros(output.length);
      var loss = 0;
      for (var i = 0; i < output.length; i++) {
        var diff = output[i] - target[i];
        loss += 0.5 * diff * diff;
        deltas[last][i] = diff * this.activationDerivative(cache.preActivations[last][i], true);
      }

      for (var l = last - 1; l >= 0; l--) {
        var layerNext = this.layers[l + 1];
        deltas[l] = zeros(this.layerSizes[l + 1]);
        for (var j = 0; j < deltas[l].length; j++) {
          var sum = 0;
          for (var o = 0; o < deltas[l + 1].length; o++) {
            sum += layerNext.weights[o][j] * deltas[l + 1][o];
          }
          deltas[l][j] = sum * this.activationDerivative(cache.preActivations[l][j], false);
        }
      }

      this.applyGradients(cache.activations, deltas);
      return loss / output.length;
    },

    applyGradients: function (activations, deltas) {
      this.t += 1;
      var opt = this.options;
      var lr = opt.learningRate;
      var clipValue = opt.gradientClip;

      for (var l = 0; l < this.layers.length; l++) {
        var layer = this.layers[l];
        var aPrev = activations[l];
        for (var o = 0; o < layer.biases.length; o++) {
          var gb = clamp(deltas[l][o], -clipValue, clipValue);
          this.updateScalar(layer, "b", o, 0, gb, lr);
          for (var j = 0; j < aPrev.length; j++) {
            var gw = deltas[l][o] * aPrev[j] + opt.l2 * layer.weights[o][j];
            gw = clamp(gw, -clipValue, clipValue);
            this.updateScalar(layer, "w", o, j, gw, lr);
          }
        }
      }
    },

    updateScalar: function (layer, kind, i, j, grad, lr) {
      var opt = this.options;
      if (opt.optimizer === "sgd") {
        if (kind === "b") layer.biases[i] -= lr * grad;
        else layer.weights[i][j] -= lr * grad;
        return;
      }

      var beta1 = opt.beta1;
      var beta2 = opt.beta2;
      var eps = opt.epsilon;
      var m;
      var v;
      if (kind === "b") {
        layer.mb[i] = beta1 * layer.mb[i] + (1 - beta1) * grad;
        layer.vb[i] = beta2 * layer.vb[i] + (1 - beta2) * grad * grad;
        m = layer.mb[i] / (1 - Math.pow(beta1, this.t));
        v = layer.vb[i] / (1 - Math.pow(beta2, this.t));
        layer.biases[i] -= lr * m / (Math.sqrt(v) + eps);
      } else {
        layer.mw[i][j] = beta1 * layer.mw[i][j] + (1 - beta1) * grad;
        layer.vw[i][j] = beta2 * layer.vw[i][j] + (1 - beta2) * grad * grad;
        m = layer.mw[i][j] / (1 - Math.pow(beta1, this.t));
        v = layer.vw[i][j] / (1 - Math.pow(beta2, this.t));
        layer.weights[i][j] -= lr * m / (Math.sqrt(v) + eps);
      }
    },

    activate: function (x, isOutput) {
      var name = isOutput ? this.options.outputActivation : this.options.activation;
      if (name === "tanh") return Math.tanh(x);
      if (name === "sigmoid") return 1 / (1 + Math.exp(-x));
      if (name === "linear") return x;
      return x > 0 ? x : 0;
    },

    activationDerivative: function (x, isOutput) {
      var name = isOutput ? this.options.outputActivation : this.options.activation;
      if (name === "tanh") {
        var y = Math.tanh(x);
        return 1 - y * y;
      }
      if (name === "sigmoid") {
        y = 1 / (1 + Math.exp(-x));
        return y * (1 - y);
      }
      if (name === "linear") return 1;
      return x > 0 ? 1 : 0;
    },

    clone: function () {
      var nn = new DenseNetwork(this.layerSizes, this.options);
      nn.fromJSON(this.toJSON());
      return nn;
    },

    copyFrom: function (other) {
      return this.fromJSON(other.toJSON());
    },

    toJSON: function () {
      return {
        layerSizes: this.layerSizes,
        options: {
          activation: this.options.activation,
          outputActivation: this.options.outputActivation,
          learningRate: this.options.learningRate,
          gradientClip: this.options.gradientClip,
          l2: this.options.l2,
          optimizer: this.options.optimizer,
          beta1: this.options.beta1,
          beta2: this.options.beta2,
          epsilon: this.options.epsilon
        },
        t: this.t,
        layers: this.layers
      };
    },

    fromJSON: function (json) {
      this.layerSizes = json.layerSizes.slice();
      this.options = defaults(json.options, this.options || {});
      this.layers = JSON.parse(JSON.stringify(json.layers));
      this.t = json.t || 0;
      return this;
    }
  };

  function DQNAgent(options) {
    options = defaults(options, {
      stateSize: 0,
      actionSize: 0,
      hiddenSizes: [32],
      gamma: 0.99,
      epsilon: 1,
      epsilonMin: 0.05,
      epsilonDecay: 0.995,
      batchSize: 32,
      replaySize: 10000,
      learningRate: 0.001,
      trainStart: 100,
      targetUpdateInterval: 100,
      doubleDQN: true,
      temporalWindow: 0,
      randomActionDistribution: null,
      rng: Math.random,
      optimizer: "adam",
      gradientClip: 10,
      l2: 0
    });

    assert(options.stateSize > 0, "stateSize is required");
    assert(options.actionSize > 0, "actionSize is required");
    if (options.randomActionDistribution) {
      assert(options.randomActionDistribution.length === options.actionSize, "randomActionDistribution length must equal actionSize");
    }

    this.options = options;
    this.stateSize = options.stateSize;
    this.actionSize = options.actionSize;
    this.temporalWindow = options.temporalWindow;
    this.inputSize = this.stateSize + this.temporalWindow * (this.stateSize + this.actionSize);
    this.rng = options.rng;
    this.replay = new ReplayBuffer(options.replaySize, this.rng);
    this.steps = 0;
    this.trainingSteps = 0;
    this.loss = 0;
    this.latestReward = 0;
    this.rewardAverage = new RollingAverage(100);
    this.lossAverage = new RollingAverage(100);
    this.lastState = null;
    this.lastAction = null;
    this.stateHistory = [];
    this.actionHistory = [];

    var sizes = [this.inputSize].concat(options.hiddenSizes).concat([this.actionSize]);
    this.online = new DenseNetwork(sizes, {
      rng: this.rng,
      learningRate: options.learningRate,
      optimizer: options.optimizer,
      gradientClip: options.gradientClip,
      l2: options.l2,
      activation: "relu",
      outputActivation: "linear"
    });
    this.target = this.online.clone();
  }

  DQNAgent.prototype = {
    act: function (state, learn) {
      if (learn === undefined) learn = true;
      this.validateState(state);
      var input = this.buildInput(state);
      var epsilon = learn ? this.options.epsilon : 0;
      var action = this.rng() < epsilon ? this.randomAction() : argmax(this.online.predict(input));

      this.lastState = copyArray(state);
      this.lastInput = input;
      this.lastAction = action;
      this.pushHistory(state, action);
      return action;
    },

    step: function (state, action, reward, nextState, done) {
      this.validateState(state);
      this.validateState(nextState);
      assert(action >= 0 && action < this.actionSize, "action out of range");

      var encodedState = this.encodeTransitionState(state);
      var encodedNext = this.buildInput(nextState);
      this.replay.push({
        state: encodedState,
        action: action,
        reward: reward,
        nextState: encodedNext,
        done: !!done
      });
      return this.learn();
    },

    learnFromReward: function (reward, nextState, done) {
      assert(this.lastState !== null && this.lastAction !== null, "Call act(state) before learnFromReward");
      return this.step(this.lastState, this.lastAction, reward, nextState, done);
    },

    learn: function () {
      this.steps += 1;
      if (this.replay.size() < this.options.trainStart || this.replay.size() < this.options.batchSize) {
        return { trained: false, loss: this.loss, epsilon: this.options.epsilon, replaySize: this.replay.size() };
      }

      var batch = this.replay.sample(this.options.batchSize);
      var totalLoss = 0;
      for (var i = 0; i < batch.length; i++) {
        var e = batch[i];
        var targetQ = copyArray(this.online.predict(e.state));
        var y = e.reward;
        if (!e.done) {
          if (this.options.doubleDQN) {
            var nextAction = argmax(this.online.predict(e.nextState));
            y += this.options.gamma * this.target.predict(e.nextState)[nextAction];
          } else {
            y += this.options.gamma * Math.max.apply(null, this.target.predict(e.nextState));
          }
        }
        targetQ[e.action] = y;
        totalLoss += this.online.trainMSE(e.state, targetQ);
      }

      this.trainingSteps += 1;
      this.loss = totalLoss / batch.length;
      this.lossAverage.add(this.loss);
      if (this.options.epsilon > this.options.epsilonMin) {
        this.options.epsilon = Math.max(this.options.epsilonMin, this.options.epsilon * this.options.epsilonDecay);
      }
      if (this.trainingSteps % this.options.targetUpdateInterval === 0) {
        this.updateTarget();
      }
      return { trained: true, loss: this.loss, epsilon: this.options.epsilon, replaySize: this.replay.size() };
    },

    observe: function (reward) {
      this.latestReward = reward;
      this.rewardAverage.add(reward);
    },

    policy: function (state) {
      this.validateState(state);
      var input = this.buildInput(state);
      var values = this.online.predict(input);
      return { action: argmax(values), values: values };
    },

    updateTarget: function () {
      this.target.copyFrom(this.online);
    },

    randomAction: function () {
      if (this.options.randomActionDistribution) {
        return sampleWeighted(this.options.randomActionDistribution, this.rng);
      }
      return Math.floor(this.rng() * this.actionSize);
    },

    pushHistory: function (state, action) {
      this.stateHistory.push(copyArray(state));
      this.actionHistory.push(action);
      while (this.stateHistory.length > this.temporalWindow) this.stateHistory.shift();
      while (this.actionHistory.length > this.temporalWindow) this.actionHistory.shift();
    },

    buildInput: function (state) {
      var out = copyArray(state);
      for (var i = this.stateHistory.length - 1; i >= 0; i--) {
        out = out.concat(this.stateHistory[i]);
        out = out.concat(this.oneHot(this.actionHistory[i]));
      }
      while (out.length < this.inputSize) out.push(0);
      return out;
    },

    encodeTransitionState: function (state) {
      if (this.lastInput && this.lastInput.length === this.inputSize) return copyArray(this.lastInput);
      return this.buildInput(state);
    },

    oneHot: function (action) {
      var out = zeros(this.actionSize);
      out[action] = 1;
      return out;
    },

    validateState: function (state) {
      assert(state && state.length === this.stateSize, "state length must equal stateSize");
    },

    stats: function () {
      return {
        steps: this.steps,
        trainingSteps: this.trainingSteps,
        epsilon: this.options.epsilon,
        replaySize: this.replay.size(),
        latestReward: this.latestReward,
        averageReward: this.rewardAverage.value(),
        loss: this.loss,
        averageLoss: this.lossAverage.value()
      };
    },

    resetEpisode: function () {
      this.lastState = null;
      this.lastInput = null;
      this.lastAction = null;
      this.stateHistory.length = 0;
      this.actionHistory.length = 0;
    },

    toJSON: function () {
      return {
        options: {
          stateSize: this.stateSize,
          actionSize: this.actionSize,
          hiddenSizes: this.options.hiddenSizes,
          gamma: this.options.gamma,
          epsilon: this.options.epsilon,
          epsilonMin: this.options.epsilonMin,
          epsilonDecay: this.options.epsilonDecay,
          batchSize: this.options.batchSize,
          replaySize: this.options.replaySize,
          learningRate: this.options.learningRate,
          trainStart: this.options.trainStart,
          targetUpdateInterval: this.options.targetUpdateInterval,
          doubleDQN: this.options.doubleDQN,
          temporalWindow: this.options.temporalWindow,
          randomActionDistribution: this.options.randomActionDistribution,
          optimizer: this.options.optimizer,
          gradientClip: this.options.gradientClip,
          l2: this.options.l2
        },
        online: this.online.toJSON(),
        target: this.target.toJSON(),
        replay: this.replay.toJSON(),
        steps: this.steps,
        trainingSteps: this.trainingSteps,
        loss: this.loss
      };
    },

    fromJSON: function (json) {
      this.options = defaults(json.options, this.options);
      this.stateSize = this.options.stateSize;
      this.actionSize = this.options.actionSize;
      this.temporalWindow = this.options.temporalWindow;
      this.inputSize = this.stateSize + this.temporalWindow * (this.stateSize + this.actionSize);
      this.online = new DenseNetwork(json.online.layerSizes, defaults({ rng: this.rng }, json.online.options));
      this.online.fromJSON(json.online);
      this.target = new DenseNetwork(json.target.layerSizes, defaults({ rng: this.rng }, json.target.options));
      this.target.fromJSON(json.target);
      this.replay = new ReplayBuffer(this.options.replaySize, this.rng).fromJSON(json.replay);
      this.steps = json.steps || 0;
      this.trainingSteps = json.trainingSteps || 0;
      this.loss = json.loss || 0;
      return this;
    }
  };

  function RollingAverage(size) {
    this.size = size;
    this.values = [];
    this.sum = 0;
  }

  RollingAverage.prototype = {
    add: function (x) {
      this.values.push(x);
      this.sum += x;
      if (this.values.length > this.size) this.sum -= this.values.shift();
    },

    value: function () {
      return this.values.length ? this.sum / this.values.length : 0;
    }
  };

  function createSeededRandom(seed) {
    var s = seed >>> 0;
    return function () {
      s += 0x6D2B79F5;
      var t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  return {
    DQNAgent: DQNAgent,
    DenseNetwork: DenseNetwork,
    ReplayBuffer: ReplayBuffer,
    RollingAverage: RollingAverage,
    argmax: argmax,
    createSeededRandom: createSeededRandom,
    version: "0.1.0"
  };
}));
