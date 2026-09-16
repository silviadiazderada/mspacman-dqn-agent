# Ms. Pac-Man DQN Agent

Class 3 assignment: train a Deep Q-Network (DQN) to play Atari Ms. Pac-Man,
choosing exploration rate, episode count, and learning rate, and reporting
the agent's learned behavior.

> **Status: notebook ready, not yet run.** The sections marked `TODO (fill
> in after running the notebook)` below get filled in with real numbers once
> `mspacman_dqn.ipynb` has been executed in Google Colab.

## The RL setup

- **Observations**: four stacked, grayscale, 84x84 game screens (stacking
  recent frames lets the network perceive motion, not just a static image).
- **Actions**: the 9 joystick moves available in Ms. Pac-Man (up/down/left/
  right, diagonals, and no-op).
- **Reward**: in-game points (dots, power pellets, and ghosts eaten while
  "powered up"), scaled down by a constant factor rather than clipped to
  -1/0/+1, so the network can still tell a 10-point dot apart from a
  200-1600 point ghost.

## Hyperparameters chosen

| Hyperparameter | Value | Why |
|---|---|---|
| Learning rate | `0.0001` | The assignment's own suggested reference value; also the standard value used in the original DQN research on Atari, so it's a well-tested starting point rather than a guess. |
| Exploration (post-warmup) | `0.1` | Anneals from 100% random down to a steady 10% floor over the first 150 episodes — enough residual randomness to keep discovering strategies, without drowning out what's been learned. |
| Episodes | `250` | Sized to fit a ~1-2 hour Colab GPU session. |

Supporting settings (replay buffer size, target-network update frequency,
warmup steps) were all scaled down roughly 10x from typical full-scale DQN
defaults, since those defaults assume millions of training frames — at this
assignment's scale, the full-scale defaults would mean the replay buffer and
target network barely engage before training ends.

## Beyond plain DQN

Per feedback that strong scores in this class have come from smarter
learning rather than just longer training, three low-complexity, well-
established upgrades were added on top of plain DQN (all validated on
Ms. Pac-Man specifically in the original published research):

- **Double DQN** — corrects a known tendency of plain DQN to overestimate
  how good its own choices are.
- **Dueling network architecture** — separately learns "how good is this
  situation" and "how much better is each move right now," which learns
  faster in states where only one or two moves actually matter (e.g., a
  ghost right next to Pac-Man).
- **Reward scaling instead of hard clipping** — preserves the real
  difference in value between small and large in-game rewards.

**Prioritized Experience Replay** (replaying the moves the network found
most surprising more often) was considered but deliberately left out of
this run — see *Limitations and next experiment* below.

## Evaluation methodology

Baseline (before training) and post-training evaluation use **identical**
settings, changing only which network is being evaluated:

- The same 5 fixed random seeds for both runs.
- 5% exploration kept during evaluation (rather than pure greedy), so a
  partially-trained network isn't penalized for occasionally getting stuck.
- A hard step cap per evaluation episode.
- Evaluation always plays the **full 3-life game** (unlike training, which
  ends each episode after the first life lost, for a clearer training
  signal) — so evaluation scores are directly comparable to the class
  leaderboard.

## Results

TODO (fill in after running the notebook):

| | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean |
|---|---|---|---|---|---|---|
| Baseline (untrained) | | | | | | |
| Trained | | | | | | |

- Episodes completed: `TODO`
- Total environment steps / learning updates performed: `TODO`
- Elapsed training time: `TODO`
- Hardware: `TODO` (expected: Google Colab, T4 GPU)

## Expected vs. observed outcome

**Expected**: at this scale — hundreds, not millions, of training episodes —
Ms. Pac-Man is a harder game than typical small classroom RL demos (a
comparable Pong demo needed roughly 900 episodes and ~7 hours to show clear
improvement, and Ms. Pac-Man's larger, multi-ghost state space is more
demanding than Pong's). The honestly expected outcome is a **modest,
possibly noisy** improvement over the random baseline — not a dramatic one.

**Observed**: `TODO` — describe what the results table above actually
showed, and whether it matched this expectation.

## Limitations and next experiment

**Limitation**: training budget (~250 episodes / 1-2 hours on a free Colab
GPU) is small relative to what full DQN research runs use, which caps how
much the agent can realistically improve in this run.

**Proposed next experiment**: add **Prioritized Experience Replay** —
instead of sampling past experience uniformly at random, replay the
transitions the network's predictions were most wrong about more often, so
learning time is spent where it's most useful. This was left out of this run
to keep the implementation and debugging surface manageable within the time
budget, but it's the most direct lever for getting more learning progress
out of the same number of episodes.

## Reproducing this

1. Open `mspacman_dqn.ipynb` in Google Colab.
2. Runtime → Change runtime type → GPU (T4).
3. Runtime → Run all.
4. All hyperparameters are set in the clearly-labeled cell near the top of
   the notebook.
