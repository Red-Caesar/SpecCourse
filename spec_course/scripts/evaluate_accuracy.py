import argparse
import subprocess
import time
from pathlib import Path

import yaml
from utils import LOG_PATH, load_config, setup_logger


def create_lm_eval_command(model_path: str, lm_eval_args: dict) -> str:
    """Create lm-eval-harness command with arguments"""
    args_str = " ".join([f"--{k} {v}" for k, v in lm_eval_args.items() if v])
    return f"lm-eval --model vllm --model_args pretrained={model_path} {args_str}"


def wait_for_session_completion(session_name: str):
    """Wait until the command in the tmux session completes"""
    while True:
        try:
            result = subprocess.run(
                ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_pid}"],
                capture_output=True,
                text=True,
            )

            if result.stdout.strip():
                pane_pid = result.stdout.strip()
                try:
                    subprocess.run(
                        ["ps", "-p", pane_pid],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True,
                    )
                    time.sleep(120)
                except subprocess.CalledProcessError:
                    break
            else:
                break
        except subprocess.CalledProcessError:
            break


def create_tmux_session(session_name: str, command: str):
    """Create a new tmux session and run the command"""
    try:
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name], stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

    subprocess.run(["tmux", "new-session", "-d", "-s", session_name])
    subprocess.run(["tmux", "send-keys", "-t", session_name, command, "C-m"])


def main():
    parser = argparse.ArgumentParser(description="Run LM evaluation in tmux sessions")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logger(LOG_PATH, "quantization")

    models = config.get("models", [])
    lm_eval_args = config.get("lm_eval_args", {})

    for model_path in models:
        session_name = Path(model_path).name.replace(".", "_").replace("-", "_")
        command = create_lm_eval_command(model_path, lm_eval_args)
        create_tmux_session(session_name, command)

        logger.info(f"Running command in tmux session: {session_name}")
        wait_for_session_completion(session_name)

        logger.info(f"Evaluation completed for {session_name}")
        logger.info("-" * 80)


if __name__ == "__main__":
    main()
