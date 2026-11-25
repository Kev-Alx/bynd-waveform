import copy

import pytest
import torch

from beyondwaveform.adam_diagnostics import counterfactual_step
from beyondwaveform.attacks import project_relative_ball_


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def fixture(steps=4, dtype=torch.float64, **options):
    delta = torch.tensor([[.03, -.04, .01], [.02, 0, .06]], dtype=dtype, requires_grad=True)
    q = torch.tensor([[2., -1., 1.], [1., .5, -2.]], dtype=dtype)
    optimizer = torch.optim.Adam([delta], lr=.07, betas=(.8, .94), eps=1e-7, **options)
    for step in range(steps):
        delta.grad = torch.tensor([[.3+step/10, -.1, .2], [-.4, .05, .2-step/20]], dtype=dtype)
        optimizer.step()
        with torch.no_grad():
            project_relative_ball_(delta, q, .1)
    return delta, q, optimizer


@pytest.mark.parametrize("steps", [0, 1, 7])
@pytest.mark.parametrize("foreach", [None, False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cloned_full_step_matches_real_next_adam_step_and_projection_bitwise(steps, foreach, dtype):
    delta, q, optimizer = fixture(steps, dtype=dtype, foreach=foreach)
    gradient = torch.tensor([[-.6, .3, .1], [.2, -.4, -.15]], dtype=delta.dtype)
    delta.grad = gradient.clone()
    original = delta.detach().clone()
    original_gradient = delta.grad
    state = copy.deepcopy(optimizer.state_dict())
    version = delta._version
    receipt = counterfactual_step(optimizer, delta, gradient, q, .1, 1e-8)
    assert torch.equal(delta, original) and delta._version == version
    assert delta.grad is original_gradient and torch.equal(delta.grad, gradient)
    assert equal_tree(optimizer.state_dict(), state)
    assert receipt["initialized_before"] == (steps > 0)
    optimizer.step()
    assert torch.equal(delta, receipt["delta_unprojected"])
    with torch.no_grad():
        project_relative_ball_(delta, q, .1, 1e-8)
    assert torch.equal(delta, receipt["delta_projected"])
    assert torch.equal(receipt["unprojected_change"], receipt["delta_unprojected"]-original)
    assert torch.equal(receipt["projected_change"], delta-original)
    assert equal_tree(optimizer.state[delta], receipt["state_after"])
    assert receipt["state_after"]["step"].item() == steps+1
    assert receipt["relative_radius_projected"] <= .100001


def test_different_gradient_uses_same_nonzero_moments_and_bias_correction():
    delta, q, optimizer = fixture(5, weight_decay=.04)
    parameter = delta.detach().clone()
    state = copy.deepcopy(optimizer.state[delta])
    group = optimizer.param_groups[0]
    asr_gradient = torch.tensor([[-.8, -.3, .7], [.1, .2, -.9]], dtype=delta.dtype)
    full_gradient = asr_gradient + .11
    full = counterfactual_step(optimizer, delta, full_gradient, q, .1, 1e-8)
    asr = counterfactual_step(optimizer, delta, asr_gradient, q, .1, 1e-8)
    assert equal_tree(full["state_before"], asr["state_before"])
    b1, b2 = group["betas"]
    g = asr_gradient + group["weight_decay"] * parameter
    m = state["exp_avg"] * b1 + (1-b1) * g
    v = state["exp_avg_sq"] * b2 + (1-b2) * g.square()
    step = state["step"].item()+1
    expected = parameter - (group["lr"]/(1-b1**step)) * m / (v.sqrt()/(1-b2**step)**.5 + group["eps"])
    torch.testing.assert_close(asr["delta_unprojected"], expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(asr["state_after"]["exp_avg"], m, atol=1e-15, rtol=1e-14)
    torch.testing.assert_close(asr["state_after"]["exp_avg_sq"], v, atol=1e-15, rtol=1e-14)
    assert not torch.equal(asr["delta_unprojected"], full["delta_unprojected"])
    assert equal_tree(optimizer.state[delta], state) and torch.equal(delta, parameter)


def test_uninitialized_state_and_absent_parameter_gradient_remain_absent():
    delta, q, optimizer = fixture(0)
    assert delta.grad is None and not optimizer.state
    gradient = torch.ones_like(delta, requires_grad=True)
    receipt = counterfactual_step(optimizer, delta, gradient, q, .1, 1e-8)
    assert delta.grad is None and gradient.grad is None and not optimizer.state
    assert receipt["state_before"] == {} and receipt["state_after"]["step"].item() == 1
    assert all(not receipt[key].requires_grad for key in
               ("delta_before", "delta_unprojected", "delta_projected", "projected_change", "unprojected_change"))
    receipt["delta_before"].zero_()
    receipt["state_after"]["exp_avg"].zero_()
    assert delta.count_nonzero() > 0 and not optimizer.state


@pytest.mark.parametrize("options", [{"amsgrad": True}, {"maximize": True}, {"amsgrad": True, "maximize": True}])
def test_nondefault_actual_group_metadata_and_state_are_preserved(options):
    delta, q, optimizer = fixture(3, **options)
    # Current LR can differ from constructor defaults, e.g. after a scheduler.
    optimizer.param_groups[0]["lr"] = .023
    optimizer.param_groups[0]["initial_lr"] = .07
    g = torch.full_like(delta, -.2)
    receipt = counterfactual_step(optimizer, delta, g, q, .1, 1e-8)
    assert receipt["param_group"]["lr"] == .023 and receipt["param_group"]["initial_lr"] == .07
    assert receipt["optimizer_defaults"]["lr"] == .07
    delta.grad = g
    optimizer.step()
    assert torch.equal(delta, receipt["delta_unprojected"])
    assert equal_tree(optimizer.state[delta], receipt["state_after"])


def test_zero_radius_projects_only_the_cloned_parameter_without_resetting_moments():
    delta, q, optimizer = fixture(2)
    state = copy.deepcopy(optimizer.state_dict())
    receipt = counterfactual_step(optimizer, delta, torch.ones_like(delta), q, 0, 1e-8)
    assert receipt["delta_projected"].count_nonzero() == 0
    assert receipt["delta_unprojected"].count_nonzero() > 0
    assert receipt["state_after"]["step"].item() == 3
    assert equal_tree(optimizer.state_dict(), state)


def test_zero_current_gradient_retains_motion_from_actual_historical_moments():
    delta, q, optimizer = fixture(4)
    before = delta.detach().clone()
    receipt = counterfactual_step(optimizer, delta, torch.zeros_like(delta), q, .1, 1e-8)
    assert receipt["initialized_before"]
    assert receipt["unprojected_change"].norm() > 0
    torch.testing.assert_close(receipt["state_after"]["exp_avg"],
                               optimizer.state[delta]["exp_avg"] * optimizer.param_groups[0]["betas"][0])
    assert torch.equal(delta, before)


def test_invalid_optimizer_layout_or_state_is_rejected_without_mutation():
    delta, q, optimizer = fixture(1)
    with pytest.raises(ValueError, match="actual torch.optim.Adam"):
        counterfactual_step(torch.optim.SGD([delta], lr=.1), delta, torch.ones_like(delta), q, .1, 1e-8)
    other = torch.zeros_like(delta, requires_grad=True)
    with pytest.raises(ValueError, match="exactly one group"):
        counterfactual_step(torch.optim.Adam([delta, other]), delta, torch.ones_like(delta), q, .1, 1e-8)
    original = delta.detach().clone()
    optimizer.state[delta]["step"] = torch.tensor(-1.)
    with pytest.raises(ValueError, match="step counter"):
        counterfactual_step(optimizer, delta, torch.ones_like(delta), q, .1, 1e-8)
    assert torch.equal(delta, original) and optimizer.state[delta]["step"].item() == -1


def test_shape_dtype_and_nonfinite_gradient_rejected_before_lazy_state_creation():
    delta, q, optimizer = fixture(0)
    for gradient in (torch.ones(1), torch.ones_like(delta, dtype=torch.float32), torch.full_like(delta, float("nan"))):
        with pytest.raises(ValueError, match="Invalid gradient"):
            counterfactual_step(optimizer, delta, gradient, q, .1, 1e-8)
        assert not optimizer.state and delta.grad is None
