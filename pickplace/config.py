"""
pickplace/config.py — every tunable constant for the hybrid pipeline.

Single source of truth. The SAC trainer, the demo generator, the LeRobot
converter, the SmolVLA trainer and the orchestrator all import from here, so a
number can never drift between "what we trained on" and "what we run".

Anything that must match the already-recorded teleop episodes (CONTROL_HZ,
RENDER_SIZE, CAMERA_NAMES, STEP_SIZE) is marked FROZEN. Changing a FROZEN value
invalidates recorded_episodes*/ and you must re-record.
"""

from pathlib import Path
import numpy as np

# ==========================================================================
# Paths
# ==========================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent

SCENE_XML = str(REPO_ROOT / "assets/franka_emika_panda/franka_pick_place_scene.xml")
TELEOP_SCENE_XML = str(REPO_ROOT / "assets/franka_emika_panda/franka_panda_scene.xml")

# The two existing teleop recording dirs, both pick-only.
TELEOP_DIRS = [
    REPO_ROOT / "recorded_episodes",
    REPO_ROOT / "recorded_episodes_fresh",
]

OUT_ROOT = REPO_ROOT / "outputs"
SCRIPTED_DEMO_DIR = OUT_ROOT / "scripted_demos"
CROPPED_TELEOP_DIR = OUT_ROOT / "cropped_teleop"
LEROBOT_DIR = OUT_ROOT / "lerobot_dataset"
SAC_CKPT_DIR = OUT_ROOT / "sac_reach"
VLA_CKPT_DIR = OUT_ROOT / "smolvla"
EVAL_DIR = OUT_ROOT / "eval"

# ==========================================================================
# Simulation  (FROZEN — must match the recorded teleop episodes)
# ==========================================================================
CONTROL_HZ = 15                       # FROZEN
RENDER_SIZE = (224, 224)              # FROZEN  (h, w)
CAMERA_NAMES = ["front_cam", "wrist_cam"]   # FROZEN
STEP_SIZE = 0.01                      # FROZEN  metres per unit action
TABLE_TOP_Z = 0.0
LIFT_THRESHOLD = 0.05

IK_DAMPING = 0.05
IK_MAX_DQ = 0.2
MAX_LEASH = 0.05          # cap on target-vs-actual EE gap (see record_episode.py)

# Cube spawn band — matches teleop/env.py reset(), so the SmolVLA grasp data
# and the SAC reach targets cover the same region.
CUBE_X_RANGE = (0.35, 0.55)
CUBE_Y_RANGE = (-0.15, 0.15)
CUBE_HALF_SIZE = 0.02

# ==========================================================================
# Task geometry
# ==========================================================================
# Fixed drop point, mirroring drop_site in franka_pick_place_scene.xml.
# Kept in sync by test_config.py, which reads the site out of the model.
DROP_POINT = np.array([0.40, 0.30, 0.01])
DROP_PLATE_HALF = 0.07
PLACE_SUCCESS_RADIUS = 0.05    # cube centre within this of DROP_POINT (xy)

PANDA_HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# ==========================================================================
# Phase handoff
# ==========================================================================
# SAC hands control to SmolVLA when the EE is within this of its goal.
# This same radius crops the teleop demos (crop_teleop.py), so the VLA's
# training distribution starts where SAC actually leaves off. Change it and
# you must re-crop AND retrain both halves.
HANDOFF_RADIUS = 0.10

# A phase that blows its budget is a failure, not a hang.
#
# The grasp budget is sized off the DATA, not off intuition. After de-idling,
# the scripted expert grasps in 76 steps and the human segments run to a p50
# of 46 and a p90 of 58, so 200 leaves roughly 2.5x headroom over the slowest
# demonstration -- enough that a slow grasp is not scored as a failed one,
# without letting a genuinely stalled policy burn the clock.
PHASE_STEP_BUDGET = {
    "reach":    120,
    "grasp":    200,
    "retract":  120,
    "transfer": 120,
    "place":    150,
}

# De-idling the teleop segments.
#
# 83% of the frames in a recorded grasp segment command ZERO motion -- median
# longest unbroken pause is 52 frames (3.5 s). That is the keyboard interface,
# not the task: the operator taps a key, then waits. Left in, those frames
# make 83% of the behaviour-cloning targets "do nothing", and the policy
# duly learns to stall.
#
# Dropping them re-times the demonstration rather than distorting it: during
# an idle frame no action is commanded and the arm barely moves, so the
# remaining (state, action) pairs still follow from one another. Measured
# result: 22,271 frames -> 4,688, p50 46 frames per grasp, which lines up with
# the scripted expert's 76.
#
# 0 = drop every idle frame outside the protected gripper window.
TELEOP_MAX_IDLE_RUN = 0

# Frames to protect after a gripper state change. These command zero motion
# but are the single most important frames in a grasp demonstration -- the
# moment the fingers close. De-idling must never remove them. Mirrors the
# scripted expert's GRIPPER_SETTLE_STEPS.
GRIPPER_PROTECT_WINDOW = 5

# Set False to train SmolVLA on scripted demonstrations alone. Kept as a flag
# because the human and scripted data have measurably different pacing; if the
# VLA dithers at rollout, this is the first thing to try.
INCLUDE_TELEOP_DATA = True

# ==========================================================================
# Action / observation spaces
# ==========================================================================
# action = [dx, dy, dz, gripper] — dx..dz in [-1, 1], scaled by STEP_SIZE.
# gripper: >= 0.5 closed, < 0.5 open. Matches the recorded teleop convention.
ACTION_DIM = 4
STATE_DIM = 8             # 7 arm joints + gripper flag  (FROZEN)

# SAC reach action is 3-D, NOT 4-D. The gripper is a given during a reach:
# open while approaching the cube, closed while transporting it. Letting RL
# relearn it only invites the policy to drop the cube mid-transfer.
SAC_ACTION_DIM = 3

# SAC reach observation, privileged (no pixels):
#   7 joint cos + 7 joint sin + 3 ee_pos + 3 goal_pos + 3 (goal - ee)
#   + 1 dist + 1 gripper flag
SAC_OBS_DIM = 7 + 7 + 3 + 3 + 3 + 1 + 1

# ==========================================================================
# SAC hyperparameters
# ==========================================================================
SAC_HIDDEN = (256, 256)
SAC_LR = 3e-4
SAC_GAMMA = 0.98          # short horizon (<=120 steps); 0.99 over-weights the tail
SAC_TAU = 0.005
SAC_BATCH_SIZE = 256
SAC_BUFFER_SIZE = 300_000
SAC_WARMUP_STEPS = 2_000  # uniform-random actions before learning starts
SAC_UPDATES_PER_STEP = 1
# Measured: the reach policy reaches 100% on all four goal clusters by step
# 20,000. 300k was the initial guess and is roughly 15x more than the task
# needs; 60k leaves comfortable margin and takes about 6 minutes.
SAC_TOTAL_STEPS = 60_000
SAC_EVAL_EVERY = 5_000
SAC_EVAL_EPISODES = 20
SAC_INIT_ALPHA = 0.1
SAC_TARGET_ENTROPY = None   # None -> -action_dim

# Reach reward shaping
REACH_MAX_STEPS = 120
REACH_DIST_WEIGHT = 1.0
REACH_SUCCESS_BONUS = 10.0
REACH_ACTION_PENALTY = 0.01
REACH_TIME_PENALTY = 0.01

# ==========================================================================
# Demo generation
# ==========================================================================
N_SCRIPTED_DEMOS = 300
SCRIPTED_NOISE_STD = 0.15      # fraction of STEP_SIZE, injected for diversity
DEMO_SEED = 0

# ==========================================================================
# SmolVLA
# ==========================================================================
HF_REPO_ID = "Prajith7roboq/franka_pick_place_hybrid"
VLM_MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SMOLVLA_BASE = "lerobot/smolvla_base"

TASK_PICK = "pick up the red cube"
TASK_PLACE = "place the red cube on the green drop point"

VLA_CHUNK_SIZE = 50
VLA_N_ACTION_STEPS = 25     # replan every 25 steps; < chunk_size so the tail
                            # of a stale chunk is never executed
VLA_BATCH_SIZE = 16
# Measured: training is GPU-bound at ~13.9 samples/s regardless of batch
# size (batch 8 -> 1.73 it/s, batch 16 -> 0.87 it/s -- identical throughput),
# so DataLoader workers do not help and num_workers stays 0. At batch 16 one
# epoch over the 27,159 frames is ~1,700 steps, about 33 minutes.
VLA_STEPS = 12_000
VLA_LR = 1e-4
VLA_BETAS = (0.9, 0.95)
VLA_WEIGHT_DECAY = 1e-10
VLA_GRAD_CLIP = 10.0
VLA_FREEZE_VISION = True
VLA_TRAIN_EXPERT_ONLY = True

# ==========================================================================
# Evaluation
# ==========================================================================
EVAL_EPISODES = 50
EVAL_SEED = 12345
