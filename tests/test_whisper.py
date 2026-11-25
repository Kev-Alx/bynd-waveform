import torch
import pytest
from types import SimpleNamespace
from whisper import Whisper, ModelDimensions
from whisper.tokenizer import get_tokenizer
from beyondwaveform.models.whisper import WhisperASR, teacher_forcing_tokens, nll_transcript, load_whisper


@pytest.mark.parametrize("policy,expected", [("original", "DON'T CHANGE USA"),
    ("lower", "don't change usa"), ("sentence", "Don't change usa")])
def test_nll_policy_changes_only_teacher_forcing_target(policy, expected):
    class Spy(torch.nn.Module):
        is_multilingual = True
        num_languages = 99
        dims = SimpleNamespace(n_text_ctx=64)

        def forward(self, mel, inputs):
            self.inputs = inputs.clone()
            return torch.zeros(1, inputs.shape[1], 51865)

    native = Spy()
    model = WhisperASR(native, nll_transcript_policy=policy)
    model.mel = lambda audio: torch.zeros(1)
    reference = "  DON'T CHANGE USA  "
    assert torch.isfinite(model.nll(torch.zeros(1000), reference))
    target_inputs, _, _ = teacher_forcing_tokens(model.tokenizer, expected, "cpu")
    assert torch.equal(native.inputs, target_inputs)
    assert reference == "  DON'T CHANGE USA  "
    assert nll_transcript(reference, policy) == expected


def test_unknown_nll_target_policy_is_rejected():
    with pytest.raises(ValueError, match="Unknown NLL transcript policy"):
        nll_transcript("hello", "unspecified")


def test_lexical_alignment_real_whisper_tokenizer():
    tokenizer = get_tokenizer(True, language="en", task="transcribe")
    inputs, labels, mask = teacher_forcing_tokens(tokenizer, "Hello world", "cpu")
    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    lexical = tokenizer.encode("Hello world")
    assert inputs[0, :len(prefix)].tolist() == prefix
    assert labels[mask].tolist() == lexical
    assert inputs[0, len(prefix)-1] == tokenizer.no_timestamps
    assert labels[0, len(prefix)-1] == lexical[0]
    assert labels[0, -1] == tokenizer.eot and not mask[0, -1]


@pytest.mark.parametrize("timestamp_policy", ["no_timestamps", "initial_timestamp"])
def test_native_frontend_and_frozen_whisper_have_input_gradients(timestamp_policy):
    torch.manual_seed(6)
    dims = ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=16, n_audio_head=2,
                           n_audio_layer=1, n_vocab=51865, n_text_ctx=64,
                           n_text_state=16, n_text_head=2, n_text_layer=1)
    native = Whisper(dims)
    # Whisper leaves decoder positions in torch.empty storage because real
    # models load them from a checkpoint. Initialize this synthetic fixture.
    torch.nn.init.normal_(native.decoder.positional_embedding, std=0.02)
    model = WhisperASR(native, nll_timestamp_policy=timestamp_policy).eval().requires_grad_(False)
    audio = (torch.randn(4000)*0.01).requires_grad_()
    loss = model.nll(audio, "hello world")
    grad, = torch.autograd.grad(loss, audio)
    assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad.norm() > 0
    assert all(p.grad is None for p in model.parameters())


def test_timestamp_conditioning_scores_lexical_tokens_only():
    tokenizer = get_tokenizer(True, language="en", task="transcribe")
    inputs, labels, mask = teacher_forcing_tokens(tokenizer, "Hello world", "cpu", "initial_timestamp")
    prefix = [*tokenizer.sot_sequence, tokenizer.timestamp_begin]
    assert inputs[0, :len(prefix)].tolist() == prefix
    assert inputs[0, len(prefix)-1] == tokenizer.timestamp_begin
    assert labels[0, len(prefix)-1] == tokenizer.encode("Hello world")[0]
    assert labels[mask].tolist() == tokenizer.encode("Hello world")
    assert not mask[0, :len(prefix)-1].any()
    assert not mask[0, -1]
    with pytest.raises(ValueError, match="Unknown NLL timestamp policy"):
        teacher_forcing_tokens(tokenizer, "Hello world", "cpu", "unknown")


def test_timestamp_policy_metadata_is_optional_and_decoding_is_unchanged(tmp_path, monkeypatch):
    import beyondwaveform.models.whisper as module

    class Native(torch.nn.Module):
        is_multilingual = True
        num_languages = 99
        dims = SimpleNamespace(n_text_ctx=64)

    checkpoint = tmp_path / "whisper.pt"
    checkpoint.write_bytes(b"test fixture")
    monkeypatch.setattr(module.whisper, "load_model", lambda *args, **kwargs: Native())
    base = {"checkpoint": str(checkpoint), "decoding": {"beam_size": 5, "temperature": 0.0}}
    old, old_meta = load_whisper(base, "cpu")
    new, new_meta = load_whisper({**base, "nll_timestamp_policy": "initial_timestamp"}, "cpu")
    assert "nll_timestamp_policy" not in old_meta
    assert new_meta == {**old_meta, "nll_timestamp_policy": "initial_timestamp"}
    calls = []
    monkeypatch.setattr(module.whisper, "transcribe", lambda model, audio, **kwargs: (calls.append(kwargs) or {"text": " hello "}))
    assert old.transcribe(torch.zeros(100)) == new.transcribe(torch.zeros(100)) == "hello"
    assert calls[0] == calls[1]
