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


# ---------------------------------------------------------------------------
# flatten_trajectories tests
#
# flatten_trajectories(Dict[agent_id, List[Trajectory]])
#   -> Dict[agent_id, {pre_move_states, actions, post_move_states_list,
#                       old_logprobs, returns, advantages}]
#
# Key invariants:
#   - All lists/tensors have length T (total transitions across all trajectories)
#   - values fed to compute_gae has T+1 elements (0.0 bootstrap appended)
#   - Trajectories are concatenated in order; episode boundaries come from dones
#   - actions: int16; old_logprobs/rewards/returns/advantages: float32
#   - done booleans are converted to float (True→1.0, False→0.0)
# ---------------------------------------------------------------------------

from .self_play import flatten_trajectories, Transition, Trajectory


def _tr(action=0, logprob=-1.0, reward=0.0, value=0.5, done=False, state_dim=4):
    """Build a minimal Transition with controllable scalar fields."""
    return Transition(
        pre_move_state=torch.zeros(state_dim),
        action_idx=action,
        logprob=logprob,
        reward=reward,
        value=value,
        done=done,
        post_move_states=(torch.zeros(state_dim),),
    )


def _traj(*transitions):
    return Trajectory(list(transitions))


def _flatten(transitions_per_agent):
    """Build and flatten trajectories. transitions_per_agent is a dict of
    agent_id -> list of Transition (put into a single Trajectory each)."""
    trajs = {aid: [_traj(*trs)] for aid, trs in transitions_per_agent.items()}
    return flatten_trajectories(trajs)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_flatten_output_keys():
    """Output dict contains exactly the expected keys for each agent."""
    data = _flatten({0: [_tr(done=True)]})
    assert set(data.keys()) == {0}
    assert set(data[0].keys()) == {
        "pre_move_states", "actions", "post_move_states_list",
        "old_logprobs", "returns", "advantages",
    }


def test_flatten_shapes_single_trajectory():
    """T transitions → length-T tensors/lists for all output fields."""
    T = 5
    trs = [_tr() for _ in range(T - 1)] + [_tr(done=True)]
    data = _flatten({0: trs})

    assert len(data[0]["pre_move_states"]) == T
    assert len(data[0]["post_move_states_list"]) == T
    assert data[0]["actions"].shape == (T,)
    assert data[0]["old_logprobs"].shape == (T,)
    assert data[0]["returns"].shape == (T,)
    assert data[0]["advantages"].shape == (T,)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

def test_flatten_dtypes():
    """actions must be int16; logprobs/returns/advantages must be float32."""
    data = _flatten({0: [_tr(action=3, logprob=-0.5, done=True)]})
    assert data[0]["actions"].dtype == torch.int16
    assert data[0]["old_logprobs"].dtype == torch.float32
    assert data[0]["returns"].dtype == torch.float32
    assert data[0]["advantages"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Field values
# ---------------------------------------------------------------------------

def test_flatten_action_and_logprob_values():
    """Scalar fields are stored in the correct positions."""
    trs = [
        _tr(action=2, logprob=-0.3, done=False),
        _tr(action=5, logprob=-1.7, done=True),
    ]
    data = _flatten({0: trs})
    assert data[0]["actions"].tolist() == [2, 5]
    torch.testing.assert_close(data[0]["old_logprobs"], t(-0.3, -1.7))


def test_flatten_done_bool_to_float():
    """done=False → 0.0, done=True → 1.0 in the dones passed to compute_gae.
    We verify indirectly: a terminal step gets returns == reward (bootstrap zeroed)."""
    r, v = 3.0, 1.5
    data = _flatten({0: [_tr(reward=r, value=v, done=True)]})
    # With done=True: delta = r + gamma*0*(1-1) - v = r-v; returns = delta+v = r
    torch.testing.assert_close(data[0]["returns"], t(r))
    torch.testing.assert_close(data[0]["advantages"], t(r - v))


def test_flatten_non_terminal_uses_bootstrap_zero():
    """The 0.0 appended as bootstrap is used when the last transition is not done.
    With a single non-terminal transition: delta = reward + gamma*0 - value = reward - value."""
    r, v = 2.0, 0.8
    data = _flatten({0: [_tr(reward=r, value=v, done=False)]})
    # values fed to GAE: [v, 0.0]; dones=[0.0]
    # delta = r + 0.99*0.0*(1-0) - v = r - v; gae = r - v (no carry, gae_prev=0)
    torch.testing.assert_close(data[0]["advantages"], t(r - v))
    torch.testing.assert_close(data[0]["returns"], t(r - v + v))  # = t(r)


# ---------------------------------------------------------------------------
# Multi-trajectory concatenation
# ---------------------------------------------------------------------------

def test_flatten_two_trajectories_concatenated():
    """Two trajectories for the same agent are concatenated in order."""
    traj1 = _traj(
        _tr(action=1, logprob=-0.5, done=False),
        _tr(action=2, logprob=-1.0, done=True),
    )
    traj2 = _traj(
        _tr(action=3, logprob=-0.2, done=False),
        _tr(action=4, logprob=-0.8, done=True),
    )
    data = flatten_trajectories({0: [traj1, traj2]})

    assert data[0]["actions"].tolist() == [1, 2, 3, 4]
    torch.testing.assert_close(data[0]["old_logprobs"], t(-0.5, -1.0, -0.2, -0.8))


def test_flatten_two_trajectories_episode_boundary():
    """GAE results from two concatenated episodes match separately-computed results.
    This confirms episode boundaries are handled correctly (same as test_episode_boundary_resets_gae
    but exercised through flatten_trajectories rather than compute_gae directly)."""
    gamma, lam = 0.99, 0.95

    # Episode 1
    traj1 = _traj(
        _tr(reward=0.0, value=0.5, done=False),
        _tr(reward=1.0, value=0.3, done=True),
    )
    # Episode 2
    traj2 = _traj(
        _tr(reward=0.0, value=0.4, done=False),
        _tr(reward=2.0, value=0.6, done=True),
    )

    data_concat = flatten_trajectories({0: [traj1, traj2]})

    # Reference: compute each episode independently
    ret1, adv1 = compute_gae(t(0.0, 1.0), t(0.5, 0.3, 0.0), t(0.0, 1.0), gamma=gamma, lam=lam)
    ret2, adv2 = compute_gae(t(0.0, 2.0), t(0.4, 0.6, 0.0), t(0.0, 1.0), gamma=gamma, lam=lam)

    torch.testing.assert_close(data_concat[0]["advantages"][:2], adv1)
    torch.testing.assert_close(data_concat[0]["advantages"][2:], adv2)
    torch.testing.assert_close(data_concat[0]["returns"][:2], ret1)
    torch.testing.assert_close(data_concat[0]["returns"][2:], ret2)


# ---------------------------------------------------------------------------
# Multiple agents
# ---------------------------------------------------------------------------

def test_flatten_multiple_agents_independent():
    """Each agent gets its own output; their data do not mix."""
    data = _flatten({
        0: [_tr(action=10, logprob=-0.1, done=True)],
        1: [_tr(action=20, logprob=-0.9, done=True)],
    })

    assert set(data.keys()) == {0, 1}
    assert data[0]["actions"].tolist() == [10]
    assert data[1]["actions"].tolist() == [20]
    torch.testing.assert_close(data[0]["old_logprobs"], t(-0.1))
    torch.testing.assert_close(data[1]["old_logprobs"], t(-0.9))
