"""Execution sandboxes. Docker is used for experiments; LocalSandbox only for scripted tests."""
import posixpath
import re
import shlex
import subprocess
from pathlib import Path

REPO = "/testbed"
ACTIVATE = "source /opt/miniconda3/bin/activate testbed"
MAX_SEARCH_HITS = 100


class LocalSandbox:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def _path(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError(f"path outside repository: {path}")
        return p

    def run(self, command: str, timeout: int = 300):
        try:
            r = subprocess.run(command, shell=True, cwd=self.root, capture_output=True, text=True,
                               errors="replace", timeout=timeout)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return 124, f"command timed out after {timeout}s"

    def read(self, path: str) -> str:
        with self._path(path).open(encoding="utf-8", errors="replace", newline="") as f:
            return f.read()

    def write(self, path: str, text: str) -> None:
        p = self._path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            f.write(text)

    def search(self, pattern: str, path: str = ".") -> str:
        rx, hits = re.compile(pattern), []
        base = self._path(path)
        for f in sorted(base.rglob("*.py") if base.is_dir() else [base]):
            if ".git" in f.parts:
                continue
            for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f.relative_to(self.root).as_posix()}:{n}:{line}")
                    if len(hits) >= MAX_SEARCH_HITS:
                        return "\n".join(hits)
        return "\n".join(hits)

    def diff(self) -> str:
        return self.run("git add -N . && git diff HEAD")[1]

    def close(self) -> None:
        pass


def repo_path(path: str) -> str:
    """Normalize a model-supplied path and refuse anything outside the repository."""
    if path.startswith(REPO + "/") or path == REPO:
        path = posixpath.relpath(path, REPO)
    norm = posixpath.normpath(path)
    if posixpath.isabs(norm) or norm == ".." or norm.startswith("../"):
        raise ValueError(f"path outside repository: {path}")
    return norm


class DockerSandbox:
    """One disposable, network-less container per run."""

    def __init__(self, image: str, docker=("docker",), memory: str = "4g"):
        self.docker = tuple(docker)
        self.cid = subprocess.run([*self.docker, "run", "-d", "--network", "none", "--memory", memory,
                                   image, "sleep", "infinity"],
                                  check=True, capture_output=True, text=True,
                                  timeout=1800).stdout.strip()  # first use may pull a multi-GB image

    def _exec(self, script: str, stdin: bytes | None = None, timeout: int = 300):
        wrapped = f"{ACTIVATE} >/dev/null 2>&1; timeout -k 5 {int(timeout)} bash -c {shlex.quote(script)}"
        try:
            r = subprocess.run([*self.docker, "exec", "-i", "-w", REPO, self.cid, "bash", "-c", wrapped],
                               input=stdin or b"", capture_output=True, timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            return 124, f"command timed out after {timeout}s"
        return r.returncode, (r.stdout + r.stderr).decode("utf-8", errors="replace")

    def run(self, command: str, timeout: int = 300):
        return self._exec(command, timeout=timeout)

    def read(self, path: str) -> str:
        code, out = self._exec(f"cat -- {shlex.quote(repo_path(path))}", timeout=30)
        if code:
            raise FileNotFoundError(out.strip())
        return out

    def write(self, path: str, text: str) -> None:
        p = shlex.quote(repo_path(path))
        code, out = self._exec(f'mkdir -p -- "$(dirname -- {p})" && cat > {p}', text.encode("utf-8"), timeout=30)
        if code:
            raise OSError(out.strip())

    def search(self, pattern: str, path: str = ".") -> str:
        script = (f"grep -rnE --include='*.py' -e {shlex.quote(pattern)} -- {shlex.quote(repo_path(path))}"
                  f" | head -n {MAX_SEARCH_HITS}")
        return self._exec(script, timeout=60)[1]

    def diff(self) -> str:
        return self._exec("git add -N . && git diff HEAD", timeout=60)[1]

    def close(self) -> None:
        subprocess.run([*self.docker, "rm", "-f", self.cid], capture_output=True, timeout=60)
