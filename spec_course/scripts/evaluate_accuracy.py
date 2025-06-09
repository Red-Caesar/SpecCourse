import argparse
import subprocess
import time
from pathlib import Path

from utils import LOG_PATH, load_config, setup_logger

logger = setup_logger(log_name="accuracy_evaluation")


def create_lm_eval_command(model_path: str, lm_eval_args: dict) -> str:
    """Create lm-eval-harness command with arguments"""
    args_str = " ".join([f"--{k} {v}" for k, v in lm_eval_args.items()])
    return f"lm-eval --model vllm --model_args pretrained={model_path} {args_str}"


def run_background_process(command: str, output_dir: Path, model_name: str) -> int:
    """Run command in background and save outputs to files"""
    console_output = output_dir / f"{model_name}_lm_evaluation.log"

    with open(console_output, "w") as process_out:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=process_out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process.pid


def wait_for_process_completion(pid: int):
    """Wait until the background process completes"""
    while True:
        try:
            subprocess.run(
                ["ps", "-p", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            time.sleep(20)
        except subprocess.CalledProcessError:
            break


def main():
    parser = argparse.ArgumentParser(description="Run LM evaluation in tmux sessions")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()
    config = load_config(args.config)

    models = config.get("models", [])
    lm_eval_args = config.get("lm_eval_args", {})

    for model_path in models:
        model_name = model_path.split("/")[-1].replace(".", "_").replace("-", "_")
        command = create_lm_eval_command(model_path, lm_eval_args)

        logger.info(f"Starting evaluation for {model_name}")
        pid = run_background_process(command, LOG_PATH, model_name)
        logger.info(f"Process started with PID: {pid}")

        wait_for_process_completion(pid)
        logger.info(f"Evaluation completed for {model_name}")
        logger.info("-" * 80)


if __name__ == "__main__":
    main()
