import torch
from whisper import Whisper, ModelDimensions
from whisper.tokenizer import get_tokenizer
from beyondwaveform.models.whisper import WhisperASR, teacher_forcing_tokens


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


def test_native_frontend_and_frozen_whisper_have_input_gradients():
    torch.manual_seed(6)
    dims = ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=16, n_audio_head=2,
                           n_audio_layer=1, n_vocab=51865, n_text_ctx=64,
                           n_text_state=16, n_text_head=2, n_text_layer=1)
    model = WhisperASR(Whisper(dims)).eval().requires_grad_(False)
    audio = (torch.randn(4000)*0.01).requires_grad_()
    loss = model.nll(audio, "hello world")
    grad, = torch.autograd.grad(loss, audio)
    assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad.norm() > 0
    assert all(p.grad is None for p in model.parameters())

