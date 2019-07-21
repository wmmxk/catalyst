import numpy as np
import torch

from .actor_critic import OnpolicyActorCritic
from catalyst.rl import utils
from catalyst.contrib.optimizers import KFACOptimizer


class A2C(OnpolicyActorCritic):
    def _init(
        self,
        use_acktr_optimizer: bool = False,
        entropy_reg_coefficient: float = 0.
    ):
        assert not use_acktr_optimizer, "Not implemented yet"
        self.use_acktr_optimizer = use_acktr_optimizer
        self.entropy_reg_coefficient = entropy_reg_coefficient

        critic_distribution = self.critic.distribution
        self._num_atoms = self.critic.num_atoms
        self._num_heads = self.critic.num_heads
        self._hyperbolic_constant = self.critic.hyperbolic_constant
        self._gammas = \
            utils.hyperbolic_gammas(
                self._gamma,
                self._hyperbolic_constant,
                self._num_heads
            )
        # 1 x num_heads x 1
        self._gammas_torch = utils.any2device(
            self._gammas, device=self._device
        )[None, :, None]

        if critic_distribution == "categorical":
            raise NotImplementedError()
        elif critic_distribution == "quantile":
            raise NotImplementedError()

    def get_rollout_spec(self):
        return {
            "action_logprob": {
                "shape": (),
                "dtype": np.float32
            },
            "done": {
                "shape": (),
                "dtype": np.bool
            },
            "return": {
                "shape": (self._num_heads, ),
                "dtype": np.float32
            },
        }

    @torch.no_grad()
    def get_rollout(self, states, actions, rewards, dones):
        assert len(states) == len(actions) == len(rewards) == len(dones)

        trajectory_len = \
            rewards.shape[0] if dones[-1] else rewards.shape[0] - 1

        states = utils.any2device(states, device=self._device)
        actions = utils.any2device(actions, device=self._device)
        rewards = np.array(rewards)[:trajectory_len]

        _, logprobs = self.actor(states, logprob=actions)
        logprobs = logprobs.cpu().numpy().reshape(-1)[:trajectory_len]

        # len x num_heads
        returns = np.stack(
            [
                utils.geometric_cumsum(gamma, rewards[:, None])[:, 0]
                for gamma in self._gammas
            ],
            axis=1
        )

        # final rollout
        dones = dones[:trajectory_len]
        assert len(returns) == len(logprobs) == len(dones)
        rollout = {
            "action_logprob": logprobs,
            "done": dones,
            "return": returns,
        }
        return rollout

    def postprocess_buffer(self, buffers, len):
        pass

    def train(self, batch, **kwargs):
        (
            states_t, actions_t, returns_t,
            states_tp1, done_t, action_logprobs_t
        ) = (
            batch["state"], batch["action"], batch["return"],
            batch["state_tp1"], batch["done"], batch["action_logprob"]
        )

        states_t = utils.any2device(states_t, device=self._device)
        actions_t = utils.any2device(actions_t, device=self._device)
        returns_t = utils.any2device(
            returns_t, device=self._device
        ).unsqueeze_(-1)
        # states_tp1 = utils.any2device(states_tp1, device=self._device)
        # done_t = utils.any2device(done_t, device=self._device)[:, None, None]

        action_logprobs_t = utils.any2device(
            action_logprobs_t, device=self._device
        )

        # critic loss
        values_tp0 = self.critic(states_t).squeeze_(dim=2)
        advantages = (returns_t - values_tp0)
        value_loss = advantages.pow(2).mean()

        # actor loss
        _, action_logprobs_tp0 = self.actor(states_t, logprob=actions_t)
        policy_loss = -(advantages.detach() * action_logprobs_tp0).mean()

        entropy = -(
            torch.exp(action_logprobs_tp0) * action_logprobs_tp0).mean()
        entropy_loss = self.entropy_reg_coefficient * entropy
        policy_loss = policy_loss + entropy_loss

        # actor update
        actor_update_metrics = self.actor_update(policy_loss) or {}

        # critic update
        critic_update_metrics = self.critic_update(value_loss) or {}

        # metrics
        kl = 0.5 * (action_logprobs_tp0 - action_logprobs_t).pow(2).mean()
        metrics = {
            "loss_actor": policy_loss.item(),
            "loss_critic": value_loss.item(),
            "kl": kl.item(),
        }
        metrics = {**metrics, **actor_update_metrics, **critic_update_metrics}
        return metrics
