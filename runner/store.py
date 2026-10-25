"""Where results go: the GCS bucket on a VM, a local folder in tests. Paths follow the bucket layout in run_config.py."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Store(Protocol):
    def exists(self, key: str) -> bool: ...
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def list(self, prefix: str) -> list[str]: ...


class LocalStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def put(self, key: str, data: bytes) -> None:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)  # never leave a half-written file that resume would mistake for a finished one

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file()) if base.exists() else []


class GCSStore:
    def __init__(self, bucket: str) -> None:
        from google.cloud import storage

        self.bucket = storage.Client().bucket(bucket)

    def exists(self, key: str) -> bool:
        return self.bucket.blob(key).exists()

    def put(self, key: str, data: bytes) -> None:
        self.bucket.blob(key).upload_from_string(data)  # single-request upload: the object appears whole or not at all

    def get(self, key: str) -> bytes:
        return self.bucket.blob(key).download_as_bytes()

    def list(self, prefix: str) -> list[str]:
        return sorted(b.name for b in self.bucket.list_blobs(prefix=prefix))


class GcloudCLIStore:
    """Same bucket through the gcloud CLI, for machines with a gcloud login but no Application Default Credentials
    (the Jetson). Slower per call; fine for scoring and tables."""

    def __init__(self, bucket: str) -> None:
        import shutil

        self.base = f"gs://{bucket}"
        self.gcloud = shutil.which("gcloud") or str(Path.home() / "google-cloud-sdk/bin/gcloud")

    def _run(self, *args: str, data: bytes | None = None) -> bytes:
        import subprocess

        return subprocess.run([self.gcloud, "storage", *args], input=data, capture_output=True, check=True).stdout

    def exists(self, key: str) -> bool:
        import subprocess

        try:
            self._run("ls", f"{self.base}/{key}")
            return True
        except subprocess.CalledProcessError:
            return False

    def put(self, key: str, data: bytes) -> None:
        self._run("cp", "-", f"{self.base}/{key}", data=data)

    def get(self, key: str) -> bytes:
        return self._run("cat", f"{self.base}/{key}")

    def list(self, prefix: str) -> list[str]:
        import subprocess

        try:
            out = self._run("ls", "-r", f"{self.base}/{prefix}**").decode()
        except subprocess.CalledProcessError:
            return []
        n = len(self.base) + 1
        return sorted(l[n:] for l in out.splitlines() if l.startswith(self.base) and not l.endswith("/"))


def open_store(target: str) -> Store:
    if not target.startswith("gs://"):
        return LocalStore(Path(target))
    try:
        import google.auth

        google.auth.default()
        return GCSStore(target[5:])
    except Exception:  # no Application Default Credentials here: use the gcloud CLI login
        return GcloudCLIStore(target[5:])
