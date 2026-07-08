import os
import shutil
import urllib.request

from config.logging_config import get_logger

logger = get_logger(__name__)

SAMPLE_URL = "https://pixelml-lab.pixelml.delivery/asd/asd-sample.mp4"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_PATH = os.path.join(_REPO_ROOT, "tests", "fixtures", "sample.mp4")
_MIN_BYTES = 1_000_000  # a real mp4, not an HTML error page
# Some CDNs (e.g. Cloudflare) return 403 for urllib's default
# "Python-urllib/x.y" User-Agent; a browser-like UA is accepted.
_USER_AGENT = "Mozilla/5.0"


def ensure_sample(path=SAMPLE_PATH, url=SAMPLE_URL):
    """Download the shared ASD test clip to `path` if not already present.

    Idempotent: a present, non-trivial file is left untouched. Downloads to a
    temp file and atomically renames, so an interrupted or truncated download
    never leaves a bad fixture in place. Returns the local path.
    """
    if os.path.exists(path) and os.path.getsize(path) >= _MIN_BYTES:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    logger.info(f"Downloading test fixture: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    if os.path.getsize(tmp) < _MIN_BYTES:
        os.remove(tmp)
        raise RuntimeError(f"downloaded fixture from {url} too small (<{_MIN_BYTES} bytes)")
    os.replace(tmp, path)
    return path
