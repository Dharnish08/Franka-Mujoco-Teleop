"""
pickplace/tests.py -- invariants that must hold for the hybrid to work.

    python -m pickplace.tests
    python -m pickplace.tests --skip-slow      # skip the SAC self-test

No pytest dependency: plain asserts and a tiny runner, so this works in the
same environment as everything else.

These are not coverage-chasing unit tests. Each one guards a specific failure
mode that would otherwise be SILENT -- the pipeline would run, produce
numbers, and be wrong. The observation-layout test is the clearest example:
if HybridController and GoalReachEnv ever disagree about what element 17 of
the observation means, SAC still returns actions and the robot still moves. It
just moves badly, and it looks like a training problem.
"""

import argparse
import sys
import traceback

import numpy as np
import mujoco

from . import config as C


_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


# ==========================================================================
@test
def test_config_matches_scene():
    """DROP_POINT in config must equal drop_site in the XML.

    These are two hand-maintained copies of one number. If they drift, the
    place-success test is measured against a point the robot was never sent
    to, and every place fails for no visible reason.
    """
    model = mujoco.MjModel.from_xml_path(C.SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "drop_site")
    assert sid >= 0, "drop_site missing from the scene"
    xml_pos = data.site_xpos[sid]
    assert np.allclose(xml_pos, C.DROP_POINT, atol=1e-6), (
        "config.DROP_POINT %s != drop_site %s" % (C.DROP_POINT, xml_pos))


@test
def test_drop_zone_clear_of_cube_spawn():
    """A cube that never moved must not be scored as placed."""
    margin_y = C.DROP_POINT[1] - C.CUBE_Y_RANGE[1]
    assert margin_y > C.PLACE_SUCCESS_RADIUS, (
        "drop point is only %.3f m from the top of the cube spawn band, "
        "within the %.3f m success radius -- a failed reach could be scored "
        "as a successful place" % (margin_y, C.PLACE_SUCCESS_RADIUS))


@test
def test_observation_layout_matches_between_train_and_rollout():
    """GoalReachEnv._obs and HybridController._reach_obs must agree EXACTLY.

    SAC is trained against one and rolled out against the other. They are
    built by different code in different files. A mismatch does not raise --
    the policy just receives a permuted vector and behaves badly, which is
    indistinguishable from undertraining. This test is the only thing
    standing between that bug and a week of confusion.
    """
    from .reach_env import GoalReachEnv
    from .orchestrator import HybridController

    env = GoalReachEnv(seed=0)
    goal = np.array([0.45, 0.10, 0.08])
    hold = 1.0
    env.reset(goal=goal, hold_gripper=hold)

    train_obs = env._obs()

    ctrl = HybridController(env.sim, reach_agent=None, vla=None)
    rollout_obs = ctrl._reach_obs(goal, hold)

    assert train_obs.shape == rollout_obs.shape == (C.SAC_OBS_DIM,), (
        "shape mismatch: train %s rollout %s expected (%d,)"
        % (train_obs.shape, rollout_obs.shape, C.SAC_OBS_DIM))
    assert np.allclose(train_obs, rollout_obs, atol=1e-6), (
        "observation layouts diverge; first difference at index %d"
        % int(np.argmax(np.abs(train_obs - rollout_obs))))
    env.close()


@test
def test_action_units_and_scale():
    """A unit action must move the EE by about STEP_SIZE, not 1 metre."""
    from .env import PickPlaceEnv

    env = PickPlaceEnv(enable_rendering=False)
    env.reset(cube_xy=(0.45, 0.0))
    before = env.get_ee_pose()[0].copy()
    for _ in range(10):
        env.apply_action([1.0, 0.0, 0.0, 0.0])
    moved = env.get_ee_pose()[0] - before

    expected = 10 * C.STEP_SIZE
    assert 0.5 * expected < moved[0] < 1.5 * expected, (
        "10 unit +x actions moved the EE %.4f m; expected about %.4f m"
        % (moved[0], expected))
    assert abs(moved[1]) < 0.02 and abs(moved[2]) < 0.02, (
        "a pure +x action produced off-axis motion %s" % moved)
    env.close()


@test
def test_action_dim_is_rejected_when_wrong():
    from .env import PickPlaceEnv

    env = PickPlaceEnv(enable_rendering=False)
    env.reset()
    for bad in ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]):
        try:
            env.apply_action(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("apply_action accepted a %d-D action" % len(bad))
    env.close()


@test
def test_grasp_predicate_needs_both_fingers_and_a_closed_gripper():
    """is_grasped must not fire on a cube merely touching the hand."""
    from .env import PickPlaceEnv

    env = PickPlaceEnv(enable_rendering=False)
    env.reset(cube_xy=(0.45, 0.0))
    assert not env.is_grasped(), "grasped at reset, before touching anything"
    assert not env.is_placed(), "placed at reset, with the cube on the table"

    # Gripper commanded closed in mid-air, nothing between the fingers.
    for _ in range(5):
        env.apply_action([0.0, 0.0, 0.0, 1.0])
    assert not env.is_grasped(), "grasped with nothing in the gripper"
    env.close()


@test
def test_place_predicate_requires_an_open_gripper():
    """Hovering the cube over the plate while holding it is not a place."""
    from .env import PickPlaceEnv
    from .scripted_expert import ScriptedExpert

    env = PickPlaceEnv(enable_rendering=False)
    env.reset(cube_xy=(0.45, 0.0))
    ex = ScriptedExpert(env, noise_std=0.0)
    assert ex.run_grasp(), "scripted grasp failed, cannot run this test"
    assert ex.transport_to_drop(), "scripted transport failed"

    # Over the plate, still holding: must NOT count.
    assert env.gripper_closed
    assert not env.is_placed(), "a held cube over the plate was scored as placed"

    assert ex.run_place(), "scripted place failed"
    assert env.is_placed(), "cube on the plate with an open gripper was not scored"
    env.close()


@test
def test_scripted_expert_succeeds():
    """The expert is the data source. If it degrades, the dataset degrades."""
    from .env import PickPlaceEnv
    from .scripted_expert import ScriptedExpert

    env = PickPlaceEnv(enable_rendering=False)
    rng = np.random.default_rng(0)
    ok = 0
    n = 10
    for _ in range(n):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ex = ScriptedExpert(env, noise_std=C.SCRIPTED_NOISE_STD, rng=rng)
        if ex.run_grasp() and ex.transport_to_drop() and ex.run_place():
            ok += 1
    assert ok >= n - 1, "scripted expert only completed %d/%d episodes" % (ok, n)
    env.close()


@test
def test_recorder_is_not_falsy_when_empty():
    """Regression: an empty recorder must still be passed through.

    SegmentRecorder defines __len__, so an empty one is falsy. The expert used
    to do `record = record or noop`, which silently replaced a live recorder
    with a no-op and produced zero-frame segments with no error anywhere.
    """
    from .env import PickPlaceEnv
    from .scripted_expert import ScriptedExpert
    from .generate_demos import SegmentRecorder

    env = PickPlaceEnv(enable_rendering=True)
    env.reset(cube_xy=(0.45, 0.0))
    rec = SegmentRecorder(env, arm_when=None)
    assert not rec, "precondition: an empty SegmentRecorder should be falsy"

    ex = ScriptedExpert(env, noise_std=0.0)
    ex.run_grasp(record=rec)
    assert len(rec) > 0, "recorder captured nothing -- the falsy-recorder bug is back"
    env.close()


@test
def test_deidle_protects_gripper_transitions():
    """De-idling must never drop the frame where the fingers close."""
    from .replay_teleop import deidle_mask

    actions = np.zeros((40, 4), dtype=np.float32)
    actions[5:10, 0] = 1.0        # some motion
    actions[20:, 3] = 1.0         # gripper closes at 20 and stays closed

    keep = deidle_mask(actions, max_idle_run=0, protect_window=5)
    assert keep[20], "the gripper-close frame was dropped"
    assert keep[21:26].all(), "the settling window after closure was dropped"
    assert keep[5:10].all(), "frames that command motion were dropped"
    assert not keep[12:19].any(), "idle frames were not dropped"


@test
def test_handoff_radius_is_reachable_by_a_scripted_controller():
    """Every phase goal must be attainable, or SAC is being asked the impossible."""
    from .reach_env import GoalReachEnv

    env = GoalReachEnv(seed=3)
    rng = np.random.default_rng(3)
    targets = {
        "cube": lambda: np.array([rng.uniform(*C.CUBE_X_RANGE),
                                  rng.uniform(*C.CUBE_Y_RANGE), 0.04]),
        "drop": lambda: C.DROP_POINT + np.array([0, 0, rng.uniform(0.08, 0.22)]),
        "home": lambda: env.home_ee_pos.copy(),
    }
    for name, sampler in targets.items():
        fails = 0
        for _ in range(8):
            g = sampler()
            env.reset(goal=g, hold_gripper=0.0)
            for _ in range(C.REACH_MAX_STEPS):
                ee = env.sim.get_ee_pose()[0]
                a = np.clip((g - ee) / C.STEP_SIZE, -1, 1)
                _, _, term, trunc, info = env.step(a)
                if term or trunc:
                    break
            fails += int(not info["is_success"])
        assert fails == 0, "%s goals unreachable in %d/8 attempts" % (name, fails)
    env.close()


@test
def test_demo_start_states_match_where_sac_hands_off():
    """The VLA must be trained on the poses SAC actually delivers.

    This is the subtlest failure in the whole pipeline and it is completely
    silent. The scripted expert used to fly its own approach: up to a hover
    waypoint directly over the cube, then straight down -- crossing the
    handoff radius at dx = -0.0001 +/- 0.0011. SAC comes in from the home pose
    and crosses it at dx = -0.0262 +/- 0.0167, a gap of 23 TRAINING standard
    deviations. On a 4cm cube that is most of the cube's width, and the VLA
    would be extrapolating ~20x beyond its data on the first frame it ever
    sees. Nothing raises; it just looks like an undertrained VLA.

    Demonstrations are therefore generated with the approach flown by SAC.
    This test asserts the two distributions still overlap.
    """
    from pathlib import Path
    import mujoco
    from .env import PickPlaceEnv

    demo_dir = Path(C.SCRIPTED_DEMO_DIR)
    ckpt = C.SAC_CKPT_DIR / "sac_reach_best.pt"
    demos = sorted(demo_dir.glob("*_grasp.npz"))[:40] if demo_dir.exists() else []
    if not demos or not ckpt.exists():
        print("(no demos or SAC checkpoint; skipping) ", end="")
        return

    env = PickPlaceEnv(enable_rendering=False)

    def ee_from_state(state):
        env.data.qpos[env._arm_qpos_ids] = state[:7]
        mujoco.mj_forward(env.model, env.data)
        return env.data.site_xpos[env._ee_site_id].copy()

    demo_off = []
    for f in demos:
        z = np.load(f, allow_pickle=True)
        if "cube_xy" not in z:
            continue
        c = z["cube_xy"]
        cube = np.array([c[0], c[1], C.TABLE_TOP_Z + C.CUBE_HALF_SIZE])
        demo_off.append(ee_from_state(z["states"][0]) - cube)
    D = np.array(demo_off)

    # Where SAC actually ends its reach, measured fresh.
    from .generate_demos import SACApproach
    approach = SACApproach(ckpt)
    rng = np.random.default_rng(11)
    sac_off = []
    for _ in range(25):
        env.reset(cube_xy=(rng.uniform(*C.CUBE_X_RANGE), rng.uniform(*C.CUBE_Y_RANGE)))
        cube = env.get_object_position()
        if approach.run(env, cube, 0.0, C.PHASE_STEP_BUDGET["reach"]):
            sac_off.append(env.get_ee_pose()[0] - cube)
    S = np.array(sac_off)
    env.close()

    assert len(S) > 10, "SAC reached the handoff in only %d/25 episodes" % len(S)

    # Compare in units of the DEMO spread: that is what the VLA can generalize
    # over. Anything past ~3 sd is extrapolation.
    gap = np.abs(S.mean(0) - D.mean(0)) / (D.std(0) + 1e-9)
    worst = int(np.argmax(gap))
    assert gap[worst] < 3.0, (
        "demo start states are %.1f sd from where SAC hands off on d%s "
        "(demos %.4f+/-%.4f, SAC %.4f). Regenerate with "
        "`python -m pickplace.generate_demos --approach sac`."
        % (gap[worst], "xyz"[worst], D.mean(0)[worst], D.std(0)[worst],
           S.mean(0)[worst]))


@test
def test_gripper_needs_settling_time_before_lifting():
    """Lifting the instant the gripper is commanded closed drops the cube.

    Measured: the fingers reach the cube in 1 control step and is_grasped()
    goes true at step 2 (0.13 s), with motion fully settling by step 7
    (0.47 s). Lifting with ZERO settle steps picks only ~10/15; one step is
    already enough for 15/15.

    This pins both ends. If GRIPPER_SETTLE_STEPS is ever dropped to 0 the
    demonstrations silently start containing failed grasps, and if the
    closure genuinely slows down (heavier object, weaker actuator) this test
    catches it rather than the dataset quietly degrading.
    """
    from .env import PickPlaceEnv
    from .scripted_expert import (ScriptedExpert, GRIPPER_SETTLE_STEPS,
                                  APPROACH_HEIGHT, GRASP_HEIGHT)

    assert GRIPPER_SETTLE_STEPS >= 1, (
        "GRIPPER_SETTLE_STEPS is %d; lifting with no settle time loses about "
        "a third of grasps" % GRIPPER_SETTLE_STEPS)

    env = PickPlaceEnv(enable_rendering=False)
    env.reset(cube_xy=(0.45, 0.0))
    ex = ScriptedExpert(env, noise_std=0.0)
    cube = env.get_object_position()
    ex._goto(cube + np.array([0, 0, APPROACH_HEIGHT]), False, 60, lambda a: None)
    ex._goto(cube + np.array([0, 0, GRASP_HEIGHT]), False, 60, lambda a: None, tol=0.008)

    steps_to_grasp = None
    for t in range(GRIPPER_SETTLE_STEPS):
        env.apply_action([0, 0, 0, 1])
        if env.is_grasped() and steps_to_grasp is None:
            steps_to_grasp = t
    assert steps_to_grasp is not None, (
        "gripper did not achieve a grasp within GRIPPER_SETTLE_STEPS=%d"
        % GRIPPER_SETTLE_STEPS)
    assert steps_to_grasp < C.GRIPPER_PROTECT_WINDOW, (
        "grasp takes %d steps but de-idling only protects %d frames after a "
        "gripper change -- the closure would be cut short in the teleop data"
        % (steps_to_grasp, C.GRIPPER_PROTECT_WINDOW))
    env.close()


@test
def test_dataset_chunks_never_cross_episode_boundaries():
    """A chunk that runs into the next episode teaches transitions that never
    happened -- and nothing would raise."""
    from .dataset import PickPlaceDataset, PACK_DIR
    from pathlib import Path

    if not (Path(PACK_DIR) / "meta.json").exists():
        print("(no pack; skipping) ", end="")
        return

    ds = PickPlaceDataset()
    rng = np.random.default_rng(0)
    for _ in range(300):
        i = int(rng.integers(0, len(ds)))
        ep = int(ds.episode_index[i])
        start, end = int(ds.starts[ep]), int(ds.ends[ep])
        assert start <= i < end, "frame %d attributed to the wrong episode" % i

        s = ds[i]
        n_real = int((~s["action_is_pad"]).sum())
        assert n_real == min(ds.chunk_size, end - i), (
            "frame %d: %d real actions, expected %d"
            % (i, n_real, min(ds.chunk_size, end - i)))
        # The real part of the chunk must equal the actual future actions.
        assert np.allclose(s["action"][:n_real], ds.actions[i:i + n_real]), (
            "chunk contents do not match the source actions at frame %d" % i)


@test
def test_collate_scales_images_into_unit_range():
    """SmolVLA's VISUAL normalization is IDENTITY -- it does NOT rescale.

    Handing it raw uint8 puts every pixel 255x out of range. The model still
    runs and still returns actions; they are just garbage. This is the exact
    failure that has no symptom other than bad behaviour.
    """
    from .dataset import PickPlaceDataset, collate, PACK_DIR
    from pathlib import Path

    if not (Path(PACK_DIR) / "meta.json").exists():
        print("(no pack; skipping) ", end="")
        return

    ds = PickPlaceDataset()
    batch = collate([ds[0], ds[1], ds[2]])

    for cam in C.CAMERA_NAMES:
        key = "observation.images.%s" % cam
        t = batch[key]
        assert t.shape == (3, 3, *C.RENDER_SIZE), (
            "%s has shape %s, expected (B, 3, H, W)" % (key, tuple(t.shape)))
        assert t.dtype.is_floating_point, "%s is %s, expected float" % (key, t.dtype)
        assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0, (
            "%s spans [%.1f, %.1f]; expected [0, 1] -- raw uint8 leaked through"
            % (key, float(t.min()), float(t.max())))
    assert batch["action"].shape == (3, C.VLA_CHUNK_SIZE, C.ACTION_DIM)
    assert len(batch["task"]) == 3 and isinstance(batch["task"][0], str)


@test
def test_dataset_contains_both_tasks():
    """One language-conditioned policy needs both skills present, or the task
    string selects nothing."""
    from .dataset import PickPlaceDataset, PACK_DIR
    from pathlib import Path

    if not (Path(PACK_DIR) / "meta.json").exists():
        print("(no pack; skipping) ", end="")
        return

    ds = PickPlaceDataset()
    assert C.TASK_PICK in ds.tasks, "pick task missing from the dataset"
    assert C.TASK_PLACE in ds.tasks, "place task missing from the dataset"
    counts = np.bincount(ds.task_index, minlength=len(ds.tasks))
    for name, n in zip(ds.tasks, counts):
        assert n > 500, "task %r has only %d frames" % (name, n)


@test
def test_diff_ik_reduces_error():
    from teleop.diff_ik import DiffIKSolver

    solver = DiffIKSolver(damping=0.05)
    rng = np.random.default_rng(0)
    J = rng.normal(size=(6, 7))
    q = np.zeros(7)
    pos = np.array([0.5, 0.0, 0.3])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    target = pos + np.array([0.01, 0.0, 0.0])

    dq = solver.solve(pos, quat, target, quat, J, q) - q
    err = np.concatenate([target - pos, np.zeros(3)])
    assert np.linalg.norm(err - J @ dq) < np.linalg.norm(err), "IK step increased error"
    assert np.allclose(solver._quat_error(quat, quat), 0.0), "nonzero error for equal quats"


@test
def test_sac_solves_pendulum():
    """Slow. If SAC is broken, do not go looking in the robot env."""
    from .sac import _selftest
    _selftest(total_steps=15_000, seed=0)


test_sac_solves_pendulum.slow = True


# ==========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-slow", action="store_true")
    p.add_argument("-k", default=None, help="run only tests whose name contains this")
    args = p.parse_args()

    selected = [t for t in _TESTS
                if (not args.k or args.k in t.__name__)
                and not (args.skip_slow and getattr(t, "slow", False))]

    passed, failed = 0, []
    for t in selected:
        name = t.__name__
        sys.stdout.write("  %-62s " % name)
        sys.stdout.flush()
        try:
            t()
        except Exception as exc:
            print("FAIL")
            failed.append((name, exc, traceback.format_exc()))
        else:
            print("ok")
            passed += 1

    print("\n%d passed, %d failed, %d skipped"
          % (passed, len(failed), len(_TESTS) - len(selected)))
    for name, exc, tb in failed:
        print("\n" + "=" * 70)
        print("FAILED: %s" % name)
        print(tb)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
