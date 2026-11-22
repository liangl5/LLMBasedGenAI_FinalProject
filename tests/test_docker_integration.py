"""Real-container check, free of LLM calls. Enable with MABENCH_DOCKER="wsl -d Ubuntu-20.04 --exec docker"."""
import os
import shlex

import pytest

from mabench.run import image_name
from mabench.sandbox import DockerSandbox
from mabench.tools import execute_tool

INSTANCE = "django__django-11099"


def test_image_name_matches_published_scheme():
    assert image_name(INSTANCE) == "swebench/sweb.eval.x86_64.django_1776_django-11099:latest"


@pytest.mark.skipif("MABENCH_DOCKER" not in os.environ, reason="MABENCH_DOCKER not set")
def test_docker_sandbox_on_real_swebench_image():
    box = DockerSandbox(image_name(INSTANCE), shlex.split(os.environ["MABENCH_DOCKER"]))
    try:
        code, out = box.run("python -c 'import django; print(django.__file__)'")
        assert code == 0 and "/testbed/django" in out
        assert box.diff() == ""

        text = 'x = "it\'s $HOME é"\n'
        assert execute_tool(box, "edit_file", {"path": "django/_probe.py", "old_text": "", "new_text": text}).startswith("created")
        assert box.read("django/_probe.py") == text
        assert "_probe.py" in box.search(r"it's \$HOME", "django")
        assert "+x = " in box.diff()
        assert execute_tool(box, "read_file", {"path": "../etc/passwd"}).startswith("ERROR")

        code, _ = box.run("python -c \"import socket; socket.create_connection(('1.1.1.1', 53), 2)\"")
        assert code != 0, "container must have no network"
        code, _ = box.run("sleep 10", timeout=1)
        assert code != 0, "timeout must stop the command"
    finally:
        box.close()
