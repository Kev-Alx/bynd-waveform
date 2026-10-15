from beyondwaveform.assets import download


def test_transparently_decompressed_response_length(tmp_path, monkeypatch):
    class Response:
        status_code = 200
        headers = {"Content-Length": "3", "Content-Encoding": "gzip"}

        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size): yield b"decompressed"

    monkeypatch.setattr("beyondwaveform.assets.requests.get", lambda *a, **kw: Response())
    output = download("https://example.org/file", tmp_path / "file")
    assert output.read_bytes() == b"decompressed"
    assert output.with_suffix(".receipt.json").exists()

