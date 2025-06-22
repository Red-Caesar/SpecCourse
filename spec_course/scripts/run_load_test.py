import argparse
import subprocess
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List

import libtmux
import requests
from tqdm import tqdm

from spec_course.scripts.utils import load_config, setup_logger

warnings.filterwarnings("ignore", category=DeprecationWarning)


logger = setup_logger(log_name="load_test_experiments")


def wait_for_server(url: str, max_retries: int = 120, delay: int = 10) -> bool:
    """Wait for the vllm server to start by checking its health endpoint"""
    for i in tqdm(range(max_retries), desc=f"Waiting for server at {url} to start..."):
        try:
            response = requests.get(url + "/health")
            if response.status_code == 200:
                time.sleep(90)  # Warm up time for the server
                return True
        except requests.RequestException:
            pass
        time.sleep(delay)
    return False


def run_background_process(
    command: str, output_dir: Path, log_name: str
) -> subprocess.Popen:
    """Run command in background and save outputs to files"""
    with open(output_dir / log_name, "w") as process_out:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=process_out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process


def save_tmux_output(session_name: str, output_dir: Path, log_name: str) -> None:
    """Save tmux session output to a file"""
    server = libtmux.Server()
    session = server.find_where({"session_name": session_name})
    if session:
        with open(output_dir / log_name, "w") as f:
            pane = session.panes[0]
            output_lines = pane.capture_pane("-S", "-10000")
            output_lines.extend(pane.capture_pane())
            output = "\n".join(line.strip("\n") for line in output_lines if line)
            f.write(output)


def create_vllm_command(setup: Dict[str, str]) -> List[str]:
    """Create vllm command with arguments"""
    server_args = setup.get("server_args", {})
    env_args = setup.get("env_args", {})
    env_args = " ".join([f"{k}={v}" for k, v in env_args.items()])
    args_str = " ".join([f"--{k} {v}" for k, v in server_args.items() if k != "model"])
    model_name = server_args["model"]
    commands = [
        "source ../.venv/bin/activate",
        f"{env_args} vllm serve {model_name} {args_str}",
    ]
    return commands


def create_load_test_command(
    load_test_args: Dict[str, str], rps: str, model_name: str, suffix_run_id: str
) -> str:
    """Create load test command with arguments"""
    args_str = " ".join(
        [f"--{k} {v}" for k, v in load_test_args.items() if k not in ["rps", "run-id"]]
    )
    full_run_id = f"{load_test_args['run-id']}_{suffix_run_id}"
    return f"python3 scripts/load_test.py {args_str} --rps {rps} --model-name {model_name} --run-id {full_run_id}"


def run_evaluation(setup: Dict[str, str]) -> None:
    """
    Run evaluation for a given setup. Steps:
    1. Start vllm server with specified model and parameters.
    2. Wait for the server to be ready.
    3. Run load test with specified RPS values.
    4. Log results and clean up.
    """
    dir_log = Path(__file__).parent.parent / ".logs"
    vllm_commands = create_vllm_command(setup["vllm"])

    model_name = (
        setup["vllm"]["server_args"]["model"].replace("/", "_").replace(".", "_")
    )
    draft_name = (
        setup["vllm"]["server_args"]
        .get("speculative_config", {})
        .get("model", "")
        .replace("/", "_")
        .replace(".", "_")
    )
    num_spec_tokens = (
        setup["vllm"]["server_args"]
        .get("speculative_config", {})
        .get("num_speculative_tokens", "")
    )
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    current_setup = f"{model_name}_{draft_name}_{num_spec_tokens}"

    log_name = f"vllm_{current_setup}_{timestamp}.log"
    logger.info("Starting vllm server")
    server = libtmux.Server()
    if server.has_session(current_setup):
        server.kill_session(current_setup)

    session = server.new_session(current_setup)
    window = session.attached_window
    pane = window.attached_pane
    for cmd in vllm_commands:
        pane.send_keys(cmd)

    base_url = "http://localhost:8000"
    if not wait_for_server(base_url):
        save_tmux_output(current_setup)
        server.kill_session(current_setup)
        raise RuntimeError(f"Server at {base_url} did not start in time")
    logger.info("Started vllm server.")

    rps = setup["load_test"]["rps"]
    if ":" in rps:
        if len(rps.split(":")) == 3:
            start, end, step = map(int, rps.split(":"))
        elif len(rps.split(":")) == 2:
            start, end = map(int, rps.split(":"))
            step = 1
        else:
            raise ValueError(
                f"Invalid RPS format: {rps}. Expected format is 'start:end:step' or 'start:end'."
            )
        rps_values = range(start, end + 1, step)
    else:
        rps_values = [rps]

    for rps_value in rps_values:
        suffix_run_id = (
            setup["vllm"]["server_args"]
            .get("speculative_config", {})
            .get("num_speculative_tokens", "")
        )
        load_test_command = create_load_test_command(
            setup["load_test"],
            rps_value,
            setup["vllm"]["server_args"]["model"],
            suffix_run_id,
        )
        log_name = f"load_test_{rps_value}_{current_setup}_{timestamp}.log"
        logger.info(f"Running load test for RPS {rps_value} with setup {current_setup}")
        load_test_process = run_background_process(load_test_command, dir_log, log_name)
        load_test_process.wait()
        logger.info(
            f"Load test completed for RPS {rps_value} and setup {current_setup}"
        )
        logger.info("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run server evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    for setup in tqdm(config["setups"]):
        try:
            run_evaluation(setup)
            logger.info("Setup completed successfully")
        except Exception as e:
            error_msg = f"Setup failed:\n{str(e)}\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            continue


if __name__ == "__main__":
    main()
