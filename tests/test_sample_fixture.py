import pytest

from utils.sample_fixture import ensure_sample


def test_ensure_sample_keeps_existing_file(tmp_path):
    dest = tmp_path / "sample.mp4"
    dest.write_bytes(b"x" * 2_000_000)  # already present and large enough
    # Bogus URL: if it tried to download it would fail, so success proves it skipped.
    out = ensure_sample(path=str(dest), url="http://invalid.invalid/x.mp4")
    assert out == str(dest)
    assert dest.stat().st_size == 2_000_000


def test_ensure_sample_downloads_when_missing(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"y" * 1_500_000)
    dest = tmp_path / "sub" / "sample.mp4"
    out = ensure_sample(path=str(dest), url=src.as_uri())  # file:// URL, no network
    assert out == str(dest)
    assert dest.exists()
    assert dest.stat().st_size == 1_500_000


def test_ensure_sample_rejects_tiny_download(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"z" * 10)  # below the sanity floor
    dest = tmp_path / "sample.mp4"
    with pytest.raises(RuntimeError):
        ensure_sample(path=str(dest), url=src.as_uri())
    assert not dest.exists()  # failed download leaves nothing behind
