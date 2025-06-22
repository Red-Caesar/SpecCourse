import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


def fetch_prompts_from_hf() -> List[str]:
    """Load prompts from huggingface dataset: mbpp"""
    from datasets import load_dataset

    dataset = load_dataset("google-research-datasets/mbpp", "full")
    return dataset["test"]["text"]


def run_k6_test(args: Dict[Any, Any]) -> None:
    """Run k6 test with provided arguments"""
    script_dir = Path(__file__).parent
    k6_script_path = script_dir / "k6_script.js"

    try:
        max_tokens_mean, max_tokens_std = map(
            int, args.output_tokens_distribution.split(",")
        )
    except Exception as e:
        raise ValueError(
            f"Invalid output tokens distribution: {args.output_tokens_distribution}. Error: {e}"
        )

    try:
        prompt_len_mean, prompt_len_std = map(
            int, args.input_tokens_distribution.split(",")
        )
    except Exception as e:
        raise ValueError(
            f"Invalid input tokens distribution: {args.input_tokens_distribution}. Error: {e}"
        )

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    current_dir = f"{args.run_id + '_' if args.run_id else ''}{timestamp}"
    results_dir = script_dir.parent / "results" / "load_test" / current_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "RPS": args.rps,
        "MODEL_NAME": args.model_name,
        "ENDPOINT_URL": args.endpoint_url,
        "API_ROUTE": "/v1/chat/completions",
        "MAX_TOKENS_MEAN": str(max_tokens_mean),
        "MAX_TOKENS_STD": str(max_tokens_std),
        "PROMPT_LEN_MEAN": str(prompt_len_mean),
        "PROMPT_LEN_STD": str(prompt_len_std),
        "DURATION": args.duration,
        "PROMPT_TYPE": args.prompt_type,
        "RESULTS_DIR": results_dir,
    }

    if args.prompt_type == "code":
        prompts = fetch_prompts_from_hf()
        prompt_file = results_dir / "prompts.json"
        with open(prompt_file, "w") as file:
            json.dump(prompts, file, indent=4)

    try:
        print("Starting proxy server")
        proxy = subprocess.Popen(
            ["node", Path(__file__).parent / "proxy_server.js"],
            env={**dict(env), **dict(subprocess.os.environ)},
        )
        time.sleep(3)

        subprocess.run(
            [
                "k6",
                "run",
                f"--summary-export={results_dir / 'metrics.json'}",
                str(k6_script_path),
            ],
            env={**dict(env), **dict(subprocess.os.environ)},
            check=True,
        )
        proxy.terminate()
        params_file = results_dir / "input_params.json"
        with open(params_file, "w") as file:
            json.dump(vars(args), file, indent=4)
    except subprocess.CalledProcessError as e:
        print(f"Error running k6 test: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run k6 load test for LLM endpoint")
    parser.add_argument("--rps", type=str, default="10", help="Requests per second")
    parser.add_argument(
        "--duration", type=str, default="10s", help="Tests duration in seconds"
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="Model name to test"
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default="http://localhost:8000",
        help="Endpoint URL",
    )
    parser.add_argument(
        "--input-tokens-distribution",
        type=str,
        default="1000,128",
        help="Generated prompt tokens distribution from <mean,std>",
    )
    parser.add_argument(
        "--output-tokens-distribution",
        type=str,
        default="256,8",
        help="Output tokens distribution from <mean,std>",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["random", "code"],
        default="random",
        help="Prompt type: random or code",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Prefix the folder name with the results of the current run.",
    )

    args = parser.parse_args()
    run_k6_test(args)


if __name__ == "__main__":
    main()
