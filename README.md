# Ms. Pac-Man DQN Agent

Class 3 assignment: train a Deep Q-Network (DQN) to play Atari Ms. Pac-Man,
choosing exploration rate, episode count, and learning rate, and reporting
the agent's learned behavior.

> **Status:** **v1** (results below) is the submitted deliverable — complete,
> consistent, and verified. A **v2** attempt (6000 episodes, adding
> Prioritized Experience Replay and N-step returns, aimed at closing more of
> the gap to the professor's ~3000 reference score) was started, but the
> Colab session disconnected overnight before training finished, and no
> reliable intermediate result was captured. v2's code and reasoning are
> still documented below as designed-but-unverified work; see *Limitations
> and next experiment* for why it didn't complete and what running it
> properly would take.

## The RL setup

- **Observations**: four stacked, grayscale, 84x84 game screens (stacking
  recent frames lets the network perceive motion, not just a static image).
- **Actions**: the 9 joystick moves available in Ms. Pac-Man (up/down/left/
  right, diagonals, and no-op).
- **Reward**: in-game points (dots, power pellets, and ghosts eaten while
  "powered up"), scaled down by a constant factor rather than clipped to
  -1/0/+1, so the network can still tell a 10-point dot apart from a
  200-1600 point ghost.

## Two iterations

**v1** (250 episodes, no Prioritized Experience Replay) reached a mean
trained score of **1070** against a baseline of **112** — a 8.6x
improvement, and better than several classmates' public results for this
same assignment. It fell well short, though, of the professor's ~3000
reference score. Digging into where that number comes from: it matches the
published **Double DQN** result for Ms. Pac-Man in the original research
(Van Hasselt et al.) — achieved after **50 million training steps**, roughly
300x more than v1 used.

*A note on how the submitted v1 numbers were produced*: v1 was originally
trained on a Colab T4 GPU (250 episodes in ~3 minutes, reaching a mean score
of 854), but the separate gameplay GIF and checkpoint files from that
session weren't downloaded before the session ended, so they couldn't be
included here. Rather than submit an executed notebook with no accompanying
GIFs, the identical v1 code (same architecture, same fixed random seeds) was
re-executed end-to-end locally on CPU to produce one complete, internally
consistent set of artifacts — notebook, GIFs, checkpoints, and plot all from
the same run. That local run reached a mean of 1070 (vs. the original run's
854) — a real, expected difference from CPU vs. GPU floating-point execution
under identical code and seeds, not a different method. 1070 is the number
reported throughout this README as the v1 result, since it's the one with a
complete, verifiable, consistent set of accompanying artifacts.

**v2** (attempted, did not complete) was designed to respond to that: it
raises `EPISODES` to 6000 (still comfortably inside the original time
budget), adds **Prioritized Experience Replay** and **N-step returns** to
make each episode's learning more efficient, and makes a small supporting
change to the exploration floor. Full reasoning for each choice is below and
in the notebook's *Hyperparameters* cell. The code was tested and verified
correct (see the commit history), but the actual overnight training run
disconnected before finishing — see *Limitations* below.

## Hyperparameters chosen (v2)

| Hyperparameter | v1 | v2 | Why changed |
|---|---|---|---|
| Learning rate | `0.0001` | `0.0001` (unchanged) | Not the bottleneck in v1 — loss was still decreasing steadily, not stuck or diverging. |
| Exploration (post-warmup) | `0.1` | `0.05` | With ~24x more training episodes, the agent gets far more chances to refine a policy, so it's worth exploiting learned behavior a bit more than a much shorter run would. A real but modest lever on its own. |
| Episodes | `250` | `6000` | The main lever. v1 used only ~3 minutes of a 1-2 hour budget. A classmate's public repo for this same assignment reported a mean trained score of 1360 using 2000 episodes; 6000 is a deliberate push past that within the same time budget. |

Supporting settings (replay buffer size, target-network update frequency,
warmup steps) were scaled ~10x down from full-scale DQN defaults in v1
(appropriate for a short run) and partially scaled back up in v2 now that
the step budget is much larger — though still far short of the ~50 million
steps the literature's ~3000 score required, which isn't feasible in a
single Colab session.

## Beyond plain DQN

Five low-complexity, well-established upgrades on top of plain DQN, all
validated on Ms. Pac-Man specifically in the original published research:

- **Double DQN** — corrects a known tendency of plain DQN to overestimate
  how good its own choices are.
- **Dueling network architecture** — separately learns "how good is this
  situation" and "how much better is each move right now," which learns
  faster in states where only one or two moves actually matter (e.g., a
  ghost right next to Pac-Man).
- **Reward scaling instead of hard clipping** — preserves the real
  difference in value between small and large in-game rewards.
- **Prioritized Experience Replay** *(new in v2)* — replays the transitions
  the network's predictions were most wrong about more often (via a sum-tree
  for efficient proportional sampling), with importance-sampling correction
  so this doesn't bias what the network learns.
- **N-step returns** *(new in v2)* — bootstraps 3 real steps ahead instead
  of 1, so the network relies more on real observed reward and less on its
  own (still-improving) guesses, which speeds up early learning.

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

**v1** (submitted tonight):

| | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean |
|---|---|---|---|---|---|---|
| Baseline (untrained) | 110.0 | 110.0 | 120.0 | 130.0 | 90.0 | **112.0** |
| Trained (250 episodes) | 880.0 | 910.0 | 900.0 | 910.0 | 1750.0 | **1070.0** |

- Episodes completed: `250`
- Total environment steps / learning updates performed: `14994 steps, 204 episodes with learning updates`
- Elapsed training time: `16.8 minutes` (local CPU re-execution; the
  original Colab GPU run took ~3 minutes — see note above)
- Hardware: local CPU (re-executed to produce consistent artifacts; original
  training was on Google Colab, T4 GPU)
- Training curves plot: embedded directly in `mspacman_dqn_v1_executed.ipynb`
  (the executed notebook has all outputs visible, including the baseline/
  trained scores printed above and this plot).
- Gameplay GIFs: [`outputs/gifs/baseline_untrained.gif`](outputs/gifs/baseline_untrained.gif),
  [`outputs/gifs/trained.gif`](outputs/gifs/trained.gif)
- Intermediate GIFs/checkpoints (every 50 episodes, per the assignment's
  25+ episode requirement): [`outputs/gifs/`](outputs/gifs/),
  [`outputs/checkpoints/`](outputs/checkpoints/)

**v2** (attempted, incomplete — no result to report): the 6000-episode run
was started on a Colab T4 GPU, reached at least episode 200 (confirmed from
a mid-run screenshot), but the session disconnected overnight before
finishing and before any post-training evaluation ran. There is no baseline/
trained score table for v2, since none was ever produced — reporting one
would mean fabricating numbers. This is itself the honest result of the v2
attempt.

## Expected vs. observed outcome

**Expected**: v1's actual result (1070, an 8.6x improvement) already beat the
honest "modest improvement" expectation I'd set going in. For v2, with 24x
more episodes plus PER and N-step, a further meaningful jump is realistic —
but matching the literature's ~3000 score exactly is not, since that
required roughly 300x the training steps this run uses. A good-faith target
is closing a substantial part of the gap, not all of it.

**Observed**: v2 never reached the observation stage — the Colab session
disconnected overnight (most likely the laptop it was tethered to went to
sleep or lost its network connection) before the 6000-episode run finished.
No score, plot, or GIF exists from that attempt to compare against v1 or the
~3000 reference. v1's 1070 remains the best verified result from this
project.

## Limitations and next experiment

**Limitation 1 (score gap)**: even v2's 6000-episode budget is a small
fraction of the ~50 million training steps the published ~3000 Double DQN
score for Ms. Pac-Man required — some of the remaining gap is very likely
explained by raw training volume alone, not something a hyperparameter or
algorithm choice can fully substitute for within one Colab session.

**Limitation 2 (v2 didn't finish)**: a free-tier Colab session depends on
the browser tab staying connected for the whole run — a laptop going to
sleep or a lost connection kills the runtime and, with it, any progress not
already checkpointed. The v2 attempt hit exactly this: it disconnected
overnight before completing, so a real, larger-scale result was never
obtained, even though the underlying code was verified correct beforehand.

**Proposed next experiment**: re-run v2 either using Colab Pro's background
execution (survives a closed tab) or in shorter, attended chunks using the
notebook's own checkpoint/resume mechanism (checkpoints save every 1000
episodes specifically so a run can pick back up rather than restart from
scratch). A secondary idea worth testing on top of that is Noisy Networks
(learned, state-dependent exploration noise instead of a hand-set epsilon
schedule), which has shown further gains on top of Double DQN + Dueling +
PER in the Rainbow paper.

## Reproducing this

- **v1** (submitted results above): `mspacman_dqn_v1_executed.ipynb` — already
  executed, all outputs visible, nothing further to run.
- **v2** (code complete, not yet successfully run to completion):
  `mspacman_dqn.ipynb` — open in Google Colab, Runtime → Change runtime type
  → GPU (T4), Runtime → Run all, and keep the tab connected for the full
  ~1-2 hour run (see *Limitations* above for why that matters). All
  hyperparameters are set in the clearly-labeled cell near the top. If
  training is interrupted, the *Resume from a checkpoint* cell picks up from
  the latest saved checkpoint rather than losing all progress.
