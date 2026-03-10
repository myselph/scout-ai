"""Tests for compute_gae() in self_play.py.

GAE algorithm (iterates in reverse from t=T-1 to 0):
  delta[t] = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
  gae      = delta[t] + gamma * lam * (1 - dones[t]) * gae
  advantages[t] = gae
  returns = advantages + values[:-1]
"""
import torch
import pytest
from .self_play import compute_gae


def t(*args):
    return torch.tensor(args, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Shape and structural invariants
# ---------------------------------------------------------------------------

def test_output_shapes():
    T = 7
    rewards = torch.zeros(T)
    values = torch.zeros(T + 1)
    dones = torch.zeros(T)
    returns, advantages = compute_gae(rewards, values, dones)
    assert returns.shape == (T,)
    assert advantages.shape == (T,)


def test_returns_equals_advantages_plus_values():
    """returns = advantages + values[:-1] must hold for any input."""
    rewards = t(0.5, 1.0, -0.2, 2.0)
    values  = t(0.4, 0.6, 0.3,  0.9, 0.0)
    dones   = t(0.0, 1.0, 0.0,  1.0)
    returns, advantages = compute_gae(rewards, values, dones)
    torch.testing.assert_close(returns, advantages + values[:-1])


# ---------------------------------------------------------------------------
# Single-step episodes
# ---------------------------------------------------------------------------

def test_single_terminal_step():
    """T=1, done=1: no bootstrap, advantage = reward - value."""
    r, v = 1.0, 0.5
    returns, advantages = compute_gae(t(r), t(v, 0.0), t(1.0))
    assert torch.allclose(advantages, t(r - v))
    assert torch.allclose(returns, t(r))


def test_single_nonterminal_step_uses_bootstrap():
    """T=1, done=0: bootstrap value is included."""
    gamma, lam = 0.99, 0.95
    r, v0, v1 = 1.0, 0.5, 0.3
    returns, advantages = compute_gae(t(r), t(v0, v1), t(0.0), gamma=gamma, lam=lam)
    expected_adv = r + gamma * v1 - v0  # single-step TD, no accumulation
    torch.testing.assert_close(advantages, t(expected_adv))
    torch.testing.assert_close(returns, t(expected_adv + v0))


# ---------------------------------------------------------------------------
# Special hyperparameter values
# ---------------------------------------------------------------------------

def test_gamma_zero_returns_equal_rewards():
    """With gamma=0 each return is just the immediate reward."""
    rewards = t(1.0, 2.0, -1.0)
    values  = t(0.3, 0.8, 0.1, 99.0)  # future values irrelevant when gamma=0
    dones   = t(0.0, 0.0, 1.0)
    returns, advantages = compute_gae(rewards, values, dones, gamma=0.0, lam=0.95)
    torch.testing.assert_close(returns, rewards)
    torch.testing.assert_close(advantages, rewards - values[:-1])


def test_lam_zero_is_one_step_td():
    """With lam=0 there is no multi-step accumulation; each advantage is the
    1-step TD error."""
    gamma, lam = 0.99, 0.0
    rewards = t(0.0, 1.0)
    values  = t(0.5, 0.3, 0.0)
    dones   = t(0.0, 1.0)

    returns, advantages = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)

    # t=1 (done): delta = 1.0 + 0 - 0.3 = 0.7; gae = 0.7 (no prior)
    # t=0 (not done): delta = 0.0 + 0.99*0.3 - 0.5; gae = delta + 0 = delta
    expected = t(
        0.0 + gamma * 0.3 - 0.5,  # t=0
        1.0 + 0.0       - 0.3,    # t=1 (done, no bootstrap)
    )
    torch.testing.assert_close(advantages, expected)


def test_perfect_value_function_zero_advantage():
    """If the value function is exact and gamma=0, advantages are zero."""
    rewards = t(1.0, 2.0, 0.5)
    values  = t(1.0, 2.0, 0.5, 0.0)  # values[t] == rewards[t]
    dones   = t(0.0, 0.0, 1.0)
    _, advantages = compute_gae(rewards, values, dones, gamma=0.0)
    torch.testing.assert_close(advantages, torch.zeros(3))


# ---------------------------------------------------------------------------
# Multi-step manual calculation
# ---------------------------------------------------------------------------

def test_two_step_manual():
    """Manually verified 2-step episode ending with done=1."""
    gamma, lam = 0.99, 0.95
    rewards = t(0.0, 1.0)
    values  = t(0.5, 0.3, 0.0)
    dones   = t(0.0, 1.0)

    returns, advantages = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)

    # t=1: delta = 1.0 + 0 - 0.3 = 0.7;  gae = 0.7
    # t=0: delta = 0.0 + 0.99*0.3 - 0.5 = -0.203
    #       gae = -0.203 + 0.99*0.95*1*0.7
    gae1 = 0.7
    delta0 = 0.0 + gamma * 0.3 - 0.5
    gae0 = delta0 + gamma * lam * gae1

    torch.testing.assert_close(advantages, t(gae0, gae1))
    torch.testing.assert_close(returns, t(gae0 + 0.5, gae1 + 0.3))


# ---------------------------------------------------------------------------
# Episode boundary / concatenated episodes
# ---------------------------------------------------------------------------

def test_episode_boundary_resets_gae():
    """done=1 must zero out the GAE carry-over into the previous timestep,
    i.e. the two episodes are independent of each other."""
    gamma, lam = 0.99, 0.95

    # Episode 1: rewards [0.0, 1.0], values [0.5, 0.3, 0.0], dones [0, 1]
    # Episode 2: rewards [0.0, 2.0], values [0.4, 0.6, 0.0], dones [0, 1]
    rewards = t(0.0, 1.0, 0.0, 2.0)
    values  = t(0.5, 0.3, 0.4, 0.6, 0.0)
    dones   = t(0.0, 1.0, 0.0, 1.0)

    returns_concat, adv_concat = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)

    # Compute each episode independently.
    ret1, adv1 = compute_gae(t(0.0, 1.0), t(0.5, 0.3, 0.0), t(0.0, 1.0), gamma=gamma, lam=lam)
    ret2, adv2 = compute_gae(t(0.0, 2.0), t(0.4, 0.6, 0.0), t(0.0, 1.0), gamma=gamma, lam=lam)

    torch.testing.assert_close(adv_concat[:2], adv1)
    torch.testing.assert_close(adv_concat[2:], adv2)
    torch.testing.assert_close(returns_concat[:2], ret1)
    torch.testing.assert_close(returns_concat[2:], ret2)


def test_consecutive_dones_are_each_independent():
    """Two adjacent terminal steps should each compute independently, with no
    shared GAE state."""
    gamma, lam = 0.99, 0.95

    rewards = t(1.0, 2.0)
    values  = t(0.5, 0.3, 0.0)
    dones   = t(1.0, 1.0)

    returns, advantages = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)

    # t=1: delta = 2.0 + 0 - 0.3 = 1.7; gae = 1.7 (done=1, no carry)
    # t=0: delta = 1.0 + 0 - 0.5 = 0.5; gae = 0.5 + gamma*lam*0*1.7 = 0.5
    torch.testing.assert_close(advantages, t(0.5, 1.7))
    torch.testing.assert_close(returns, t(1.0, 2.0))  # 0.5+0.5, 1.7+0.3


# ---------------------------------------------------------------------------
# All-zeros edge cases
# ---------------------------------------------------------------------------

def test_zero_rewards_and_zero_values():
    """All zeros → advantages and returns are zero."""
    T = 5
    returns, advantages = compute_gae(
        torch.zeros(T), torch.zeros(T + 1), torch.zeros(T)
    )
    assert torch.all(advantages == 0)
    assert torch.all(returns == 0)


# ---------------------------------------------------------------------------
# ppo_loss tests
#
# PPO clipped surrogate objective:
#   ratio        = exp(log π_new - log π_old)
#   policy_loss  = -sum(min(ratio*A, clip(ratio, 1-ε, 1+ε)*A)) / minibatch_size
#   value_loss   = vf_coef * sum((returns - V)²) / minibatch_size
#   entropy_loss = ent_coef * sum(H) / minibatch_size
#   total        = policy_loss + value_loss - entropy_loss
# ---------------------------------------------------------------------------

from .self_play import ppo_loss


class MockAgent:
    """Returns fixed tensors from the agent interface used by ppo_loss."""

    def __init__(self, logprobs: torch.Tensor, entropies: torch.Tensor,
                 values: torch.Tensor):
        self._logprobs = logprobs
        self._entropies = entropies
        self._values = values

    def compute_logprobs_and_entropy_batched(self, post_move_states_list, action_indices):
        return self._logprobs, self._entropies

    def compute_values(self, pre_move_states):
        return self._values


def _dummy_inputs(batch_size=4, state_dim=8):
    """Minimal dummy inputs whose contents are irrelevant (MockAgent ignores them)."""
    pre = [torch.zeros(state_dim) for _ in range(batch_size)]
    post = [(torch.zeros(state_dim),) for _ in range(batch_size)]
    act = torch.zeros(batch_size, dtype=torch.int16)
    return pre, post, act


def test_ppo_policy_loss_ratio_one():
    """When old == new logprobs the ratio is 1 everywhere and no clipping fires.
    policy_loss = -sum(advantages) / minibatch_size."""
    B = 4
    pre, post, act = _dummy_inputs(B)
    advantages = t(1.0, -2.0, 3.0, -0.5)
    logprob = t(-1.0, -2.0, -0.5, -1.5)

    agent = MockAgent(logprob.clone(), torch.zeros(B), torch.zeros(B))
    total, metrics = ppo_loss(
        agent, pre, post, act,
        old_logprob=logprob, returns=torch.zeros(B), advantages=advantages,
        clip_ratio=0.2, vf_coef=0.0, ent_coef=0.0, minibatch_size=B,
    )

    expected = -advantages.sum() / B
    torch.testing.assert_close(total, expected)


def test_ppo_policy_loss_clips_large_ratio_positive_advantage():
    """ratio >> 1+clip_ratio and A > 0: min picks clipped term → effective ratio = 1+ε."""
    B = 2
    pre, post, act = _dummy_inputs(B)
    clip_ratio = 0.2
    advantages = t(2.0, 3.0)
    # exp(-1 - (-5)) = exp(4) ≈ 54  >> 1.2
    old_logprob = t(-5.0, -5.0)
    new_logprobs = t(-1.0, -1.0)

    agent = MockAgent(new_logprobs, torch.zeros(B), torch.zeros(B))
    total, _ = ppo_loss(
        agent, pre, post, act,
        old_logprob=old_logprob, returns=torch.zeros(B), advantages=advantages,
        clip_ratio=clip_ratio, vf_coef=0.0, ent_coef=0.0, minibatch_size=B,
    )

    expected = -(1 + clip_ratio) * advantages.sum() / B
    torch.testing.assert_close(total, expected, atol=1e-5, rtol=0)


def test_ppo_policy_loss_clips_small_ratio_negative_advantage():
    """ratio << 1-clip_ratio and A < 0: clipped term (ratio→1-ε) is more negative,
    so min picks it → effective ratio = 1-ε."""
    B = 2
    pre, post, act = _dummy_inputs(B)
    clip_ratio = 0.2
    advantages = t(-2.0, -3.0)
    # exp(-5 - (-1)) = exp(-4) ≈ 0.018  << 0.8
    old_logprob = t(-1.0, -1.0)
    new_logprobs = t(-5.0, -5.0)

    agent = MockAgent(new_logprobs, torch.zeros(B), torch.zeros(B))
    total, _ = ppo_loss(
        agent, pre, post, act,
        old_logprob=old_logprob, returns=torch.zeros(B), advantages=advantages,
        clip_ratio=clip_ratio, vf_coef=0.0, ent_coef=0.0, minibatch_size=B,
    )

    expected = -(1 - clip_ratio) * advantages.sum() / B
    torch.testing.assert_close(total, expected, atol=1e-5, rtol=0)


def test_ppo_no_clipping_within_trust_region():
    """ratio ∈ [1-ε, 1+ε]: unclipped term applies and loss = -sum(ratio*A)/B."""
    B = 3
    pre, post, act = _dummy_inputs(B)
    advantages = t(1.0, -1.0, 2.0)
    old_logprob = t(-1.0, -1.0, -1.0)
    # exp(0.05) ≈ 1.051, within [0.8, 1.2]
    new_logprobs = t(-0.95, -0.95, -0.95)

    agent = MockAgent(new_logprobs, torch.zeros(B), torch.zeros(B))
    total, _ = ppo_loss(
        agent, pre, post, act,
        old_logprob=old_logprob, returns=torch.zeros(B), advantages=advantages,
        clip_ratio=0.2, vf_coef=0.0, ent_coef=0.0, minibatch_size=B,
    )

    ratio = torch.exp(new_logprobs - old_logprob)
    expected = -(ratio * advantages).sum() / B
    torch.testing.assert_close(total, expected, atol=1e-5, rtol=0)


def test_ppo_value_loss_mse():
    """value_loss = vf_coef * sum((returns - predicted_values)²) / B."""
    B = 4
    pre, post, act = _dummy_inputs(B)
    returns = t(1.0, 2.0, 3.0, 4.0)
    pred_values = t(1.5, 1.5, 3.5, 3.5)
    vf_coef = 0.5

    # Zero out policy and entropy components
    agent = MockAgent(torch.zeros(B), torch.zeros(B), pred_values)
    total, metrics = ppo_loss(
        agent, pre, post, act,
        old_logprob=torch.zeros(B), returns=returns, advantages=torch.zeros(B),
        clip_ratio=0.2, vf_coef=vf_coef, ent_coef=0.0, minibatch_size=B,
    )

    expected = vf_coef * ((returns - pred_values) ** 2).sum() / B
    torch.testing.assert_close(total, expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(
        torch.tensor(metrics["value_loss"]), expected, atol=1e-5, rtol=0)


def test_ppo_entropy_bonus_reduces_total_loss():
    """Higher entropy → lower total loss (entropy_bonus is subtracted)."""
    B = 2
    pre, post, act = _dummy_inputs(B)
    zeros = torch.zeros(B)

    def run(entropy_val):
        agent = MockAgent(zeros.clone(), torch.full((B,), entropy_val), zeros.clone())
        total, _ = ppo_loss(
            agent, pre, post, act,
            old_logprob=zeros, returns=zeros, advantages=zeros,
            clip_ratio=0.2, vf_coef=0.0, ent_coef=0.01, minibatch_size=B,
        )
        return total

    assert run(1.0) < run(0.1), "Higher entropy should produce a lower total loss"


def test_ppo_total_loss_is_sum_of_components():
    """total = policy_loss + value_loss - entropy_bonus, exactly."""
    B = 3
    pre, post, act = _dummy_inputs(B)
    advantages = t(1.0, 0.5, -0.5)
    old_logprob = t(-1.0, -1.0, -1.0)
    new_logprobs = t(-1.0, -1.0, -1.0)   # ratio = 1
    entropies = t(0.5, 0.3, 0.4)
    returns = t(2.0, 1.0, 0.0)
    pred_values = t(1.5, 1.2, 0.3)
    vf_coef, ent_coef = 0.5, 0.01

    agent = MockAgent(new_logprobs, entropies, pred_values)
    total, metrics = ppo_loss(
        agent, pre, post, act,
        old_logprob=old_logprob, returns=returns, advantages=advantages,
        clip_ratio=0.2, vf_coef=vf_coef, ent_coef=ent_coef, minibatch_size=B,
    )

    policy_loss = -advantages.sum() / B
    value_loss  = vf_coef * ((returns - pred_values) ** 2).sum() / B
    entropy_bonus = ent_coef * entropies.sum() / B
    expected = policy_loss + value_loss - entropy_bonus

    torch.testing.assert_close(total, expected, atol=1e-5, rtol=0)


def test_ppo_metrics_keys_and_entropy_value():
    """Metrics dict has exactly {policy_loss, value_loss, entropy} and entropy
    is the mean entropy over the batch."""
    B = 4
    pre, post, act = _dummy_inputs(B)
    entropies = t(0.6, 0.4, 0.8, 0.2)
    agent = MockAgent(torch.zeros(B), entropies, torch.zeros(B))

    _, metrics = ppo_loss(
        agent, pre, post, act,
        old_logprob=torch.zeros(B), returns=torch.zeros(B),
        advantages=torch.zeros(B),
        clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01, minibatch_size=B,
    )

    assert set(metrics.keys()) == {"policy_loss", "value_loss", "entropy"}
    assert metrics["entropy"] == pytest.approx(entropies.mean().item(), abs=1e-5)


def test_ppo_minibatch_size_scales_loss():
    """Doubling minibatch_size halves all loss components because each term is
    divided by minibatch_size."""
    B = 4
    pre, post, act = _dummy_inputs(B)
    advantages = torch.ones(B)
    entropies = torch.full((B,), 0.5)
    returns = torch.ones(B)
    pred_values = torch.zeros(B)

    def run(mb):
        agent = MockAgent(torch.zeros(B), entropies.clone(), pred_values.clone())
        total, _ = ppo_loss(
            agent, pre, post, act,
            old_logprob=torch.zeros(B), returns=returns, advantages=advantages,
            clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01, minibatch_size=mb,
        )
        return total

    loss_B = run(B)
    loss_2B = run(B * 2)
    torch.testing.assert_close(loss_B, loss_2B * 2, atol=1e-5, rtol=0)
