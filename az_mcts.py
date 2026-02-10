import math
import numpy as np
import torch
from az_model import obs_to_tensor


class Node:
    def __init__(self, prior):
        self.prior = float(prior)
        self.visit = 0
        self.value_sum = 0.0
        self.children = {}

    def value(self):
        if self.visit == 0:
            return 0.0
        return self.value_sum / self.visit


class MCTS:
    def __init__(
        self,
        model,
        action_size,
        sims=64,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        device=None,
        dt_ms=16,
    ):
        self.model = model
        self.action_size = action_size
        self.sims = sims
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = device
        self.dt_ms = dt_ms

    def run(self, env):
        root = Node(0.0)
        self._expand(root, env, add_noise=True)

        for _ in range(self.sims):
            node = root
            sim_env = env.clone()
            path = [node]

            while node.children:
                action, node = self._select_child(node)
                sim_env.step_action(action, dt_ms=self.dt_ms)
                path.append(node)
                if sim_env.is_terminal():
                    break

            if sim_env.is_terminal():
                value = sim_env.terminal_value()
            else:
                value = self._expand(node, sim_env, add_noise=False)

            for n in path:
                n.visit += 1
                n.value_sum += value

        policy = self._policy_from_root(root)
        return policy

    def _select_child(self, node):
        best_score = -1e9
        best_action = None
        best_child = None
        parent_visits = max(1, node.visit)
        for action, child in node.children.items():
            q = child.value()
            u = self.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _expand(self, node, env, add_noise=False):
        planes, _, _ = env.get_observation()
        with torch.no_grad():
            obs = obs_to_tensor(planes, self.device)
            logits, value = self.model(obs)
        logits = logits.squeeze(0).detach().cpu().numpy()
        mask = np.asarray(env.legal_action_mask(), dtype=np.float32)
        priors = self._softmax_masked(logits, mask)

        if add_noise:
            valid = mask > 0
            if valid.any():
                noise = np.random.dirichlet([self.dirichlet_alpha] * valid.sum())
                priors[valid] = (1 - self.dirichlet_eps) * priors[valid] + self.dirichlet_eps * noise

        for action, prior in enumerate(priors):
            if prior > 0:
                node.children[action] = Node(prior)

        return float(value.item())

    def _policy_from_root(self, root):
        counts = np.zeros(self.action_size, dtype=np.float32)
        for action, child in root.children.items():
            counts[action] = child.visit
        if counts.sum() <= 0:
            return np.ones(self.action_size, dtype=np.float32) / self.action_size
        return counts / counts.sum()

    @staticmethod
    def _softmax_masked(logits, mask):
        logits = np.asarray(logits, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if mask.sum() <= 0:
            return np.ones_like(logits, dtype=np.float32) / len(logits)
        masked = np.where(mask > 0, logits, -1e9)
        max_logit = np.max(masked)
        exp = np.exp(masked - max_logit) * mask
        total = exp.sum()
        if total <= 0:
            return mask / mask.sum()
        return exp / total
