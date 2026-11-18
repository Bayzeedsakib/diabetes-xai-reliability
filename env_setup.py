"""Environment guard for every thesis notebook from Stage 3 on (S10).

Standard library only, so it can run before numpy, pandas or scikit-learn are imported.
Call it in the first code cell:

    import env_setup
    env_setup.ensure_pinned(PROJECT_DIR, install=IN_COLAB)

- On Colab (install=True): every package pinned in requirements.txt that is missing or at another
  version is installed at the pinned version. If a package that was already installed changed
  version, the notebook stops and asks for a runtime restart, because the old version may already
  be loaded.
- On the PC (install=False): the notebook stops if any pinned package is missing or differs, and
  prints the pip command to fix it.
"""
import os
import subprocess
import sys
from importlib import metadata


def read_pins(project_dir):
    path = os.path.join(project_dir, "requirements.txt")
    if not os.path.exists(path):
        raise SystemExit(f"requirements.txt not found in {project_dir}. Copy the canonical Colab file (S6) there.")
    pins = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "==" in line:
                name, ver = line.split("==", 1)
                pins[name.strip()] = ver.strip()
    return pins


def installed_version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def pin_differences(project_dir):
    """{package: (pinned, installed or None)} for every pinned package that is missing or differs."""
    return {p: (v, installed_version(p)) for p, v in read_pins(project_dir).items() if installed_version(p) != v}


def ensure_pinned(project_dir, install=False):
    diff = pin_differences(project_dir)
    if not diff:
        print("All pinned packages present at the pinned versions.")
        return {}
    if not install:
        lines = "\n".join(f"  {p}: pinned {want}, installed {have or 'MISSING'}" for p, (want, have) in diff.items())
        raise SystemExit("Environment differs from requirements.txt (S6/S10):\n" + lines +
                         "\nFix with: pip install -r requirements.txt   then restart the kernel.")
    changed = [p for p, (_, have) in diff.items() if have is not None]   # version swap, not a fresh install
    specs = [f"{p}=={want}" for p, (want, _) in diff.items()]
    print("Installing pinned versions:", " ".join(specs))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *specs])
    still = pin_differences(project_dir)
    if still:
        raise SystemExit(f"Still different after install: {still}")
    if changed:
        raise SystemExit("Pinned versions installed over different ones: " + ", ".join(changed) +
                         ". Restart the runtime (Runtime > Restart session) and run all again.")
    print("Missing pinned packages installed; no restart needed.")
    return diff
