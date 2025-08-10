import gymnasium as gym
from typing import Dict, Optional, Tuple
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
import torch
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)
from ray.rllib.utils.typing import (
    TensorType,
)
from typing import Tuple, Optional, Dict

class ActionMaskingRLModuleMulti(ActionMaskingTorchRLModule):

    @override(ActionMaskingTorchRLModule)
    def _preprocess_batch(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        """Extracts observations and action mask from the batch

        Args:
            batch: A dictionary containing tensors (at least `Columns.OBS`)

        Returns:
            A tuple with the action mask tensor and the modified batch containing
                the original observations.
        """
        # Check observation specs for action mask and observation keys.
        self._check_batch(batch)
        # Extract the available actions tensor from the observation.
        agent_id = next(iter(batch[Columns.OBS])) 
        action_mask = batch[Columns.OBS][agent_id].pop("action_mask")

        # Modify the batch for the `DefaultPPORLModule`'s `forward` method, i.e.
        # pass only `"obs"` into the `forward` method.
        batch[Columns.OBS] = batch[Columns.OBS][agent_id].pop("observations")

        # Return the extracted action mask and the modified batch.
        return action_mask, batch
    
    @override(ActionMaskingTorchRLModule)
    def _check_batch(self, batch: Dict[str, TensorType]) -> Optional[ValueError]:
            """Assert that the batch includes action mask and observations.

            Args:
                batch: A dicitonary containing tensors (at least `Columns.OBS`) to be
                    checked.

            Raises:
                `ValueError` if the column `Columns.OBS`  does not contain observations
                    and action mask.
            """
            if not self._checked_observations:
                agent_id = next(iter(batch[Columns.OBS])) 
                if "action_mask" not in batch[Columns.OBS][agent_id]:
                    raise ValueError(
                        "No action mask found in observation. This `RLModule` requires "
                        "the environment to provide observations that include an "
                        "action mask (i.e. an observation space of the Dict space "
                        "type that looks as follows: \n"
                        "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                        "'observations': self.observation_space}"
                    )
                if "observations" not in batch[Columns.OBS][agent_id]:
                    raise ValueError(
                        "No observations found in observation. This 'RLModule` requires "
                        "the environment to provide observations that include the original "
                        "observations under a key `'observations'` in a dict (i.e. an "
                        "observation space of the Dict space type that looks as follows: \n"
                        "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                        "'observations': <observation_space>}"
                    )
                self._checked_observations = True

class ActionMaskingRLModuleMultiRandom(ActionMaskingTorchRLModule):

    """Masks and also makes the dist"""
    @override(ActionMaskingTorchRLModule)
    def _mask_action_logits(
        self, batch: Dict[str, TensorType], action_mask: TensorType
    ) -> Dict[str, TensorType]:
        uniform_logits = torch.where(action_mask.bool(), torch.ones_like(action_mask), torch.zeros_like(action_mask))
        masked_logits = torch.log(uniform_logits / uniform_logits.sum(dim=1, keepdim=True))
        # Mask the logits.
        batch[Columns.ACTION_DIST_INPUTS] = masked_logits

        # Return the batch with the masked action logits.
        return batch

    @override(ActionMaskingTorchRLModule)
    def _preprocess_batch(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        """Extracts observations and action mask from the batch

        Args:
            batch: A dictionary containing tensors (at least `Columns.OBS`)

        Returns:
            A tuple with the action mask tensor and the modified batch containing
                the original observations.
        """
        # Check observation specs for action mask and observation keys.
        self._check_batch(batch)
        # Extract the available actions tensor from the observation.
        agent_id = next(iter(batch[Columns.OBS])) 
        action_mask = batch[Columns.OBS][agent_id].pop("action_mask")

        # Modify the batch for the `DefaultPPORLModule`'s `forward` method, i.e.
        # pass only `"obs"` into the `forward` method.
        batch[Columns.OBS] = batch[Columns.OBS][agent_id].pop("observations")

        # Return the extracted action mask and the modified batch.
        return action_mask, batch
    
    @override(ActionMaskingTorchRLModule)
    def _check_batch(self, batch: Dict[str, TensorType]) -> Optional[ValueError]:
            """Assert that the batch includes action mask and observations.

            Args:
                batch: A dicitonary containing tensors (at least `Columns.OBS`) to be
                    checked.

            Raises:
                `ValueError` if the column `Columns.OBS`  does not contain observations
                    and action mask.
            """
            if not self._checked_observations:
                agent_id = next(iter(batch[Columns.OBS])) 
                if "action_mask" not in batch[Columns.OBS][agent_id]:
                    raise ValueError(
                        "No action mask found in observation. This `RLModule` requires "
                        "the environment to provide observations that include an "
                        "action mask (i.e. an observation space of the Dict space "
                        "type that looks as follows: \n"
                        "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                        "'observations': self.observation_space}"
                    )
                if "observations" not in batch[Columns.OBS][agent_id]:
                    raise ValueError(
                        "No observations found in observation. This 'RLModule` requires "
                        "the environment to provide observations that include the original "
                        "observations under a key `'observations'` in a dict (i.e. an "
                        "observation space of the Dict space type that looks as follows: \n"
                        "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                        "'observations': <observation_space>}"
                    )
                self._checked_observations = True

class ActionMaskingRLModuleMultiRuleBased(ActionMaskingTorchRLModule):

    """Masks and also makes the dist"""
    @override(ActionMaskingTorchRLModule)
    def _mask_action_logits(
        self, batch: Dict[str, TensorType], action_mask: TensorType
    ) -> Dict[str, TensorType]:
        uniform_logits = torch.where(action_mask.bool(), torch.ones_like(action_mask), torch.zeros_like(action_mask))
        masked_logits = torch.log(uniform_logits / uniform_logits.sum(dim=1, keepdim=True))
        # Mask the logits.
        batch[Columns.ACTION_DIST_INPUTS] = masked_logits

        # Return the batch with the masked action logits.
        return batch

    @override(ActionMaskingTorchRLModule)
    def _preprocess_batch(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        """Extracts observations and action mask from the batch

        Args:
            batch: A dictionary containing tensors (at least `Columns.OBS`)

        Returns:
            A tuple with the action mask tensor and the modified batch containing
                the original observations.
        """
        # Check observation specs for action mask and observation keys.
        self._check_batch(batch)
        # Extract the available actions tensor from the observation.
        agent_id = next(iter(batch[Columns.OBS])) 
        action_mask = batch[Columns.OBS][agent_id].pop("action_mask")

        # Modify the batch for the `DefaultPPORLModule`'s `forward` method, i.e.
        # pass only `"obs"` into the `forward` method.
        batch[Columns.OBS] = batch[Columns.OBS][agent_id].pop("observations")

        # Return the extracted action mask and the modified batch.
        return action_mask, batch
    
    @override(ActionMaskingTorchRLModule)
    def _check_batch(self, batch: Dict[str, TensorType]) -> Optional[ValueError]:
            """Assert that the batch includes action mask and observations.

            Args:
                batch: A dicitonary containing tensors (at least `Columns.OBS`) to be
                    checked.

            Raises:
                `ValueError` if the column `Columns.OBS`  does not contain observations
                    and action mask.
            """
            if not self._checked_observations:
                agent_id = next(iter(batch[Columns.OBS])) 
                if "action_mask" not in batch[Columns.OBS][agent_id]:
                    raise ValueError(
                        "No action mask found in observation. This `RLModule` requires "
                        "the environment to provide observations that include an "
                        "action mask (i.e. an observation space of the Dict space "
                        "type that looks as follows: \n"
                        "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                        "'observations': self.observation_space}"
                    )
                if "observations" not in batch[Columns.OBS][agent_id]:
                    raise ValueError(
                        "No observations found in observation. This 'RLModule` requires "
                        "the environment to provide observations that include the original "
                        "observations under a key `'observations'` in a dict (i.e. an "
                        "observation space of the Dict space type that looks as follows: \n"
                        "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                        "'observations': <observation_space>}"
                    )
                self._checked_observations = True