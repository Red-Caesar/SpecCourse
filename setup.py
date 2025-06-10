import os
import shutil
import subprocess
from pathlib import Path

from setuptools import Command, find_packages, setup

ROOT_DIR = Path(__file__).parent


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / "requirements"

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif (
                not line.startswith("--")
                and not line.startswith("#")
                and line.strip() != ""
            ):
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("dev.txt")

    return requirements


def install_nvm_and_node():
    """Install nvm and Node.js if not already installed"""
    home = Path.home()
    nvm_dir = home / ".nvm"

    if not nvm_dir.exists():
        print("Installing nvm...")
        try:
            subprocess.run(
                "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash",
                shell=True,
                check=True,
            )

            subprocess.run(
                ["bash", "-c", ". ~/.nvm/nvm.sh && nvm install 20 && nvm use 20"],
                check=True,
            )
            print("nvm and Node.js installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error installing nvm: {e}")
            return False
    else:
        print("nvm is already installed")
    return True


def install_k6():
    """Install k6 from binary release"""
    k6_version = "v0.58.0"

    if shutil.which("k6"):
        print("k6 is already installed")
        return True

    try:
        print("Installing k6...")
        bin_dir = Path.home() / ".local" / "bin"

        subprocess.run(
            [
                "bash",
                "-c",
                f"curl -L https://github.com/grafana/k6/releases/download/{k6_version}/k6-{k6_version}-linux-amd64.tar.gz | tar xz && mv k6-{k6_version}-linux-amd64/k6 {bin_dir}/k6 && rm -rf k6-{k6_version}-linux-amd64",
            ],
            check=True,
        )

        os.environ["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
        print("k6 installed successfully")
        return True

    except Exception as e:
        print(f"Error installing k6: {e}")
        return False


def install_lm_eval():
    """Install lm-evaluation-harness from deps"""
    try:
        print("Installing lm-eval...")
        lm_eval_dir = ROOT_DIR / "deps" / "lm-evaluation-harness"

        if not lm_eval_dir.exists():
            print(f"Error: Directory not found: {lm_eval_dir}")
            return False

        subprocess.run(["uv", "pip", "install", "-e", "."], cwd=lm_eval_dir, check=True)
        print("lm-eval installed successfully")
        return True
    except Exception as e:
        print(f"Error installing lm-eval: {e}")
        return False


class PostInstallCommand(Command):
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        install_nvm_and_node()
        install_k6()
        install_lm_eval()


setup(
    name="spec_course",
    packages=find_packages(where="spec_course"),
    cmdclass={
        "install": PostInstallCommand,
    },
    python_requires=">=3.12",
)
