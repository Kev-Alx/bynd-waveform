from pathlib import Path
import sys

import torch
from beyondwaveform.config import AttackConfig, SpectralConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_adam_contributions import compare_branches, measure_trajectory


def test_local_comparison_uses_actual_moments_and_reports_projected_displacement():
    q = torch.ones(2, 2, dtype=torch.float64)*10
    delta = torch.zeros_like(q, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=.01)
    delta.grad = torch.tensor([[1., -2.], [3., -4.]], dtype=torch.float64)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = delta.detach().clone()
    gradients = {"full": torch.tensor([[2., -3.], [4., -5.]], dtype=torch.float64),
        "negative_nll": torch.ones_like(q), "weighted_tv": torch.zeros_like(q), "weighted_hf": torch.ones_like(q)}
    branches, rows = compare_branches(optimizer, delta, gradients, q)
    assert torch.equal(delta, before) and delta.grad is None
    assert rows["full"]["projected_difference_from_full_norm"] == 0
    assert rows["without_tv"]["projected_difference_from_full_norm"] == 0
    assert rows["asr_only"]["projected_difference_from_full_norm"] > 0
    delta.grad = gradients["full"]
    optimizer.step()
    torch.testing.assert_close(delta, branches["full"]["delta_projected"], rtol=0, atol=0)
    assert optimizer.state[delta]["step"] == 2


def test_full_diagnostic_advances_only_fifty_steps_and_saves_four_moment_receipts(tmp_path):
    class Vocoder(torch.nn.Module):
        def forward(self, features):
            return torch.tanh(features.flatten().repeat_interleave(4))

    class Surrogate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def nll(self, audio, text):
            self.calls += 1
            return (audio+.2).square().mean()

    torch.manual_seed(112)
    q = torch.randn(4, 8, dtype=torch.float64)
    x = torch.randn(128, dtype=torch.float64)*.1
    surrogate = Surrogate()
    result = measure_trajectory(q, x, "FIXED", Vocoder(), surrogate, AttackConfig(learning_rate=.1),
        SpectralConfig(n_fft=8, hop_length=2, win_length=8), tmp_path)
    assert surrogate.calls == result["actual_optimization_updates"] == len(result["trace"]) == 50
    assert result["counterfactual_trajectory_continuations"] == 0
    assert [row["step"] for row in result["captures"]] == [1, 10, 25, 50]
    assert all(row["full_clone_max_abs_error"] == 0 for row in result["captures"])
    assert all(row["radius_after_update"] <= .100001 for row in result["trace"])
    for step in result["captures"]:
        saved = torch.load(step["evidence_path"], weights_only=True)
        assert torch.equal(saved["actual_delta_after"], saved["branches"]["full"]["delta_projected"])
        assert set(saved["branches"]) == {"full", "asr_only", "without_tv", "without_hf"}
