# Hybrid SAC + SmolVLA pick-and-place

Franka Panda pick-and-place in MuJoCo, split between a reinforcement-learned
reach policy and a vision-language-action manipulation policy.

The division of labour is the point: **RL moves the arm through free space,
the VLA does everything that involves touching the cube.** Free-space reaching
has a dense, well-shaped reward and needs no demonstrations, so RL is cheap
there. Grasping and placing are contact-rich and semantically specified, which
is what a VLA is for.

## The pipeline

| # | Phase | Controller | Goal / task | Advances when |
|---|----------|-----------|--------------------------------|------------------------------|
| 1 | reach | SAC | cube position | EE within `HANDOFF_RADIUS` |
| 2 | grasp | SmolVLA | `"pick up the red cube"` | cube held **and** lifted |
| 3 | retract | SAC | home EE pose, gripper closed | EE within `HANDOFF_RADIUS` |
| 4 | transfer | SAC | above the drop plate | EE within `HANDOFF_RADIUS` |
| 5 | place | SmolVLA | `"place the red cube …"` | cube on plate, gripper open |

**One** goal-conditioned SAC policy serves phases 1, 3 and 4 — the goal is an
input, not a separate network. **One** language-conditioned SmolVLA serves
phases 2 and 5 — the task string selects the skill. Training two of either
would throw away the thing that makes each approach worth using.

## Results

50 episodes, identical seeded cube positions, SmolVLA at step 12000.

| | scripted ablation | real SmolVLA |
|---|---|---|
| success | **100.0%** (50/50) | **92.0%** (46/50) |
| place error | 0.9 cm | 1.3 cm |
| steps | 148 | 155 |

Funnel for the real VLA: reach 50/50, grasp 49/50, retract 47/50,
transfer 46/50, place 46/50. The ablation is what makes that readable -- the
scripted stand-in scoring 100% says SAC, both handoffs, the phase machine and
the task definition are all sound, so the entire 8-point gap is the VLA.

And it is not a grasp-RATE problem: 49 of 50 episodes grasped. Three of the
four failures are the cube slipping in transit afterwards, i.e. grasp
PRECISION. VLA grasps are 2.5x less well centred and 3.7x more variable
vertically than the demonstrations they learned from, and `is_picked()` --
held and lifted, a binary -- passes them anyway.

### How many training steps does this need?

Not the 12,000 it was run for. Success vs step, 50 episodes each:

| step | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 | 10000 | 11000 | 12000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| success | 32% | 52% | 86% | 88% | 78% | 72% | 100% | 90% | 80% | 88% | 88% | 92% |

Steps 3000-12000 average **86.2%** (range 72-100%, sd 7.5%) with no trend --
the curve plateaus at ~3000 and the remaining 8,000 steps buy nothing
measurable. Note the spread exceeds binomial sampling error (4.9% at n=50),
so checkpoints genuinely differ; do not read a trend off adjacent points.
`eval_checkpoints.py` reports "7000 would have been enough", which anchors on
the maximum and overstates it -- 3000-4000 is the honest recommendation.

SmolVLA is also STOCHASTIC at inference: its flow-matching head samples fresh
noise per chunk, and the same checkpoint on the same cube positions scored
29/30 and 21/30 on runs differing only in prior RNG state. Both evaluators
seed torch for this reason. Any VLA measurement taken without that is
unreproducible.

### Highest-value next step

Grasp precision, not more training. Either tighten the demonstration success
criterion beyond `is_picked()`, or lower `n_action_steps` -- a grasp is ~40
frames, so at the trained default of 25 the policy re-plans about once and is
otherwise open-loop. A preliminary sweep held 27/27 through transport at
n_action_steps=10 versus 17/21 at 25; worth confirming properly, since it is
an inference-time change requiring no retraining.

## Running it

```bash
# 0. one-time environment setup (see "Environment" below)

# 1. train the reach policy            ~6 min, reaches 100% by step 20k
python -m pickplace.train_reach

# 2. generate manipulation demonstrations
python -m pickplace.generate_demos --n 300      # scripted, both segments
python -m pickplace.replay_teleop               # re-render your teleop episodes

# 3. pack the training set                       ~19 s, lossless
python -m pickplace.dataset --pack

# 4. fine-tune SmolVLA
python -m pickplace.train_smolvla

# optional: publish the same episodes to the Hub as a LeRobotDataset
python -m pickplace.to_lerobot --overwrite --push

# 5. evaluate
python -m pickplace.evaluate --vla scripted     # pipeline ablation first
python -m pickplace.evaluate                    # the real hybrid

# invariants
python -m pickplace.tests
```

## What each file does

| File | Role |
|---|---|
| `config.py` | Every constant. Values marked FROZEN must match the recorded teleop episodes. |
| `env.py` | `PickPlaceEnv` — the drop zone, the task predicates, and `apply_action`. |
| `reach_env.py` | Gymnasium env for the SAC half. Goal-conditioned, privileged state, no pixels. |
| `sac.py` | Self-contained SAC. `python -m pickplace.sac` self-tests on Pendulum-v1. |
| `train_reach.py` | Trains the reach policy; reports success **per goal cluster**. |
| `scripted_expert.py` | Privileged waypoint expert. The demonstration source. |
| `generate_demos.py` | Records grasp and place segments, armed at the handoff condition. |
| `crop_teleop.py` | Finds the grasp segment in a recorded teleop episode. |
| `replay_teleop.py` | Replays teleop actions in the new scene and re-renders them. |
| `dataset.py` | Packs the episodes into memmapped arrays; serves action chunks. **The training path.** |
| `to_lerobot.py` | Optional `LeRobotDataset` export, for publishing to the Hub. Not required to train. |
| `train_smolvla.py` | Fine-tunes SmolVLA on both skills at once. |
| `vla_policy.py` | Inference wrapper. Owns the image/action/task conversions. |
| `orchestrator.py` | The five-phase state machine. This is the hybrid. |
| `evaluate.py` | Success rate **as a funnel**, plus the scripted ablation. |
| `tests.py` | Invariants whose violation would otherwise be silent. |

## Design decisions worth knowing

**One shared action interface.** SAC, the scripted expert, SmolVLA and the
replay pipeline all drive the robot through `PickPlaceEnv.apply_action`, which
reproduces `record_episode.py`'s control loop exactly — persistent EE target,
leash clamp, damped least-squares IK. An action therefore means the same thing
to every consumer. A policy trained against one action semantics and rolled
out under another fails in ways that look like a training problem.

**SAC does not control the gripper.** Its action is 3-D. The gripper is open
during reach and closed during retract/transfer because the cube is in hand —
there is no decision to make, so it is not modelled as one.

**LeRobotDataset is not on the training path.** Fine-tuning SmolVLA needs only
two things a plain array does not give you free: normalization statistics, and
action chunks with a pad mask. Both are a few lines. `LeRobotDataset` supplies
them but also re-encodes every frame to h264 and decodes it back at training
time — a *lossy* round trip on pristine synthetic renders, taking the better
part of an hour for 695 episodes, and its batched-encoding path is broken in
0.4.4. `dataset.py` packs the same episodes into memmapped `.npy` in **19
seconds**, losslessly. `to_lerobot.py` remains for publishing to the Hub.

**Evaluate the pipeline before blaming the VLA.** `--vla scripted` swaps
SmolVLA for the scripted expert and changes nothing else. If the hybrid scores
well there and badly with the real VLA, the gap is the VLA. If it scores badly
both ways, the problem is upstream and no amount of VLA training will fix it.

## Things that were measured, not assumed

These were all found by instrumenting rather than reasoning, and each one
changed the design:

- **The demonstrations began in poses SAC never produces.** The scripted
  expert flew its own approach — up to a hover waypoint directly over the cube,
  then straight down — crossing the handoff radius at `dx = -0.0001 ± 0.0011`.
  SAC arrives from the home pose and crosses it at `dx = -0.0262 ± 0.0167`:
  **23 training standard deviations**. On a 4cm cube that is most of the cube's
  width, and since the demos carry ~1mm of lateral variation anywhere, the VLA
  would have been extrapolating ~20× beyond its data on the first frame it ever
  saw. Nothing errors — it just looks like an undertrained VLA, which is
  exactly why the scripted ablation exists. Fixed by flying the demo approach
  with the trained SAC policy (`--approach sac`, now the default), which brings
  the gap to 0.6σ. Worth noting the 98 replayed teleop episodes were already
  fine here: human keyboard approaches land at `dx = +0.0071 ± 0.0353`, a
  spread 2.1× wider than SAC's that *covers* its operating point.

- **The cube was slipping out of the gripper during transport** — 26 of 40
  episodes lost between grasp and drop point. Neither `panda.xml` nor the
  original scene sets any friction, so both the cube and the fingertip pads
  fell back to MuJoCo defaults (`condim=3`, no torsional friction). Adding
  `condim="4" friction="1.6 0.05 0.001"` to the cube took the scripted expert
  from 35% to **100%** end-to-end.

- **83% of every recorded teleop grasp segment commands zero motion.** Median
  longest unbroken pause is 52 frames (3.5 s) — that is keyboard teleop, not
  the task. Left in, 83% of the behaviour-cloning targets would be "do
  nothing" and the policy learns to stall. De-idling cuts 22,271 frames to
  4,688 with a median of 46 per grasp, in line with the scripted expert's 76.

- **The IK marker was blocking the wrist camera.** `ee_site` is declared
  `rgba="0 1 0 1"` and sits 10 cm down the tool axis, filling **13.5%** of
  every recorded wrist frame, dead centre, exactly where the cube belongs
  during a grasp. It is needed for IK, so it is hidden at render time via the
  scene option rather than deleted.

- **The teleop episodes are replayable.** They record actions, and actions are
  all the simulator needs — so they were re-driven through the new scene and
  re-rendered, which fixed both the marker and the missing drop plate without
  costing the human trajectories. 96% replay successfully with a median
  terminal joint error of 0.0055 rad; the rest are discarded, not trusted.

- **Goals high and close to the base are unreachable** with the wrist
  quaternion held fixed, regardless of radius — a scripted controller misses
  `(0.25,-0.30,0.55)` at radius 0.68 while hitting `(0.60,+0.40,0.55)` at
  radius 0.91. The goal sampler is capped accordingly.

- **Start poses beyond radius 0.84 cannot be recovered from.** The arm is
  near-singular radially and damped least-squares cannot pull it back. Every
  one of 5 stuck cases in 120 was past that radius. Bounding the start-state
  randomization removed a ~4% unrecoverable floor.

- **SAC converges far faster than budgeted** — 100% on all four goal clusters
  by step 20,000, not the 300,000 originally configured.

## Environment

Verified on Windows 11, RTX 5070 Ti Laptop (12.8 GB, sm_120), Python 3.11.

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install "transformers>=4.57.1,<5.0.0" "huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36.0"
pip install num2words peft tensorboard
```

Two version constraints matter:

- **`transformers` must stay on 4.x.** On 5.x, lerobot 0.4.4 fails at import —
  its groot policy hits `non-default argument 'backbone_cfg' follows default
  argument` while building a dataclass, which takes `lerobot.policies` down
  with it.
- **`torch` must be a cu128 build.** The RTX 5070 Ti is Blackwell (sm_120) and
  the default PyPI wheel is CPU-only. lerobot pins `torch<2.11`; 2.11 works in
  practice and was verified by running a full SmolVLA forward/backward pass,
  not by trusting the pin.

`batch_encoding_size` must stay at 1 when building the dataset — see the note
in `to_lerobot.py`.

## Measured resource use

| | |
|---|---|
| Env step, no rendering | ~1,900 steps/s |
| Env step, 2 cameras | ~465 steps/s |
| SAC training | ~180 steps/s, 100% by 20k steps |
| SmolVLA | 450M params, 100M trainable; 1.6 GB VRAM at batch 2 |
