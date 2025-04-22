import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict


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

    env = {
        "RPS": args.rps,
        "MODEL_NAME": args.model_name,
        "ENDPOINT_URL": args.endpoint_url,
        "MAX_TOKENS_MEAN": max_tokens_mean,
        "MAX_TOKENS_STD": max_tokens_std,
        "PROMPT_LEN_MEAN": prompt_len_mean,
        "PROMPT_LEN_STD": prompt_len_std,
        "DURATION": args.duration,
    }

    try:
        subprocess.run(
            ["k6", "run", str(k6_script_path)],
            env={**dict(env), **dict(subprocess.os.environ)},
            check=True,
        )
    finally:
        k6_script_path.unlink(missing_ok=True)


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
        default="http://localhost:8000/v1/chat/completions",
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

    args = parser.parse_args()
    run_k6_test(args)


if __name__ == "__main__":
    main()
