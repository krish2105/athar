"""Register a repository-local Jupyter kernel pointing at this project's venv.

`make notebooks` executes the notebooks headlessly, which needs a kernel whose
interpreter has ATHAR and its dependencies importable. The machine's global
`python3` kernelspec points wherever it happens to point — on a shared laptop,
often at a different project's virtual environment, or at one that no longer
exists.

Writing the kernelspec into `.jupyter/` inside the repository, and pointing
`JUPYTER_PATH` at it from the Makefile, makes `make all` behave the same on a
fresh clone as it does here without editing anything outside the project.
"""

import json
import sys
from pathlib import Path

KERNEL = "athar"


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / ".jupyter" / "kernels" / KERNEL
    directory.mkdir(parents=True, exist_ok=True)
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "ATHAR",
        "language": "python",
    }
    (directory / "kernel.json").write_text(json.dumps(spec, indent=1) + "\n")
    print(f"registered kernel {KERNEL!r} at {directory} -> {sys.executable}")


if __name__ == "__main__":
    main()
