"""
pickplace/vla_policy.py -- SmolVLA at rollout time.

A thin wrapper whose entire job is to make the boundary between the simulator
and the policy unambiguous. Three conversions live here and nowhere else:

  IMAGES   PickPlaceEnv renders uint8 HWC in [0, 255]. SmolVLA wants float32
           CHW in [0, 1], and its VISUAL normalization is IDENTITY -- meaning
           it does NOT rescale for you. Feeding raw uint8 through would put
           every pixel 255x out of range and the policy would emit garbage
           while looking perfectly healthy.

  ACTIONS  The policy emits normalized [-1, 1] deltas, which is exactly what
           apply_action expects. No conversion -- but asserted, because this
           is the interface where a silent unit mismatch would be most costly.

  TASK     The natural-language string selects the skill. Switching it MUST be
           accompanied by policy.reset(): SmolVLA buffers a chunk of
           n_action_steps future actions, and without a reset the first steps
           of the place phase would execute leftover actions planned for the
           grasp.
"""

import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)

from . import config as C


class SmolVLAController:

    def __init__(self, checkpoint_dir, device=None, n_action_steps=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint_dir)

        # How many actions to execute before re-planning. This is an INFERENCE
        # knob -- the model predicts a chunk of chunk_size actions and
        # n_action_steps of them get executed before it looks at the world
        # again. Lower is more closed-loop and costs proportionally more
        # forward passes.
        #
        # It matters here: a whole grasp is only ~40 frames, so at the trained
        # default of 25 the policy re-plans roughly once during the entire
        # grasp and is otherwise flying blind. Measured against the scripted
        # expert, VLA grasps are 2.5x less well centred and 3.7x more variable
        # vertically -- which is what later slips during transport.
        if n_action_steps is not None:
            self.policy.config.n_action_steps = int(n_action_steps)

        self.policy.to(self.device)
        self.policy.eval()

        self.pre = PolicyProcessorPipeline.from_pretrained(
            checkpoint_dir, config_filename="policy_preprocessor.json"
        )
        # The two converters MUST be passed explicitly on load. They are
        # plain functions, so they are not serialized into the pipeline JSON,
        # and from_pretrained silently falls back to the default dict-shaped
        # converter. The postprocessor is handed a raw action Tensor, so that
        # default dies with "EnvTransition must be a dictionary. Got Tensor"
        # -- at the first VLA step of the first rollout, long after training.
        self.post = PolicyProcessorPipeline.from_pretrained(
            checkpoint_dir,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        self._task = None

    # ----------------------------------------------------------------------
    def set_task(self, task):
        """Switch skill. Always clears the action queue -- see the module note."""
        if task != self._task:
            self._task = task
            self.policy.reset()

    def reset(self):
        self.policy.reset()

    # ----------------------------------------------------------------------
    def _observation(self, obs):
        out = {
            "observation.state": torch.from_numpy(
                np.asarray(obs["state"], dtype=np.float32)
            ),
            "task": self._task,
        }
        for cam, img in obs["images"].items():
            arr = np.asarray(img)
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            # HWC -> CHW
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = np.transpose(arr, (2, 0, 1))
            out["observation.images.%s" % cam] = torch.from_numpy(
                np.ascontiguousarray(arr, dtype=np.float32)
            )
        return out

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs):
        if self._task is None:
            raise RuntimeError("set_task() before act()")
        batch = self.pre(self._observation(obs))
        action = self.post(self.policy.select_action(batch))
        a = action.squeeze(0).float().cpu().numpy()

        if a.shape[0] != C.ACTION_DIM:
            raise ValueError("policy returned %d-D action, expected %d"
                             % (a.shape[0], C.ACTION_DIM))
        # Clip rather than trust. A flow-matching head is a regressor: nothing
        # in it guarantees the output stays inside the training range.
        a[:3] = np.clip(a[:3], -1.0, 1.0)
        a[3] = 1.0 if a[3] >= 0.5 else 0.0
        return a.astype(np.float32)
