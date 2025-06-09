import argparse
import contextlib
import gc
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import ray
import torch
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

from spec_course.scripts.utils import load_config, setup_logger

os.environ["VLLM_USE_V1"] = "0"

logger = setup_logger(log_name="sd_experiments")


@dataclass
class SDMetrics:
    main_model: str
    speculative_model: Optional[str]
    dataset_type: str
    num_prompts: int
    time_taken: float
    timestamp: str
    mean_acceptance_length: Optional[float] = None
    acceptance_rates: Optional[List[float]] = None

    def to_dict(self):
        return asdict(self)

    def save_to_json(self, output_path: Path):
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def prepare_prompts(dataset_type: str, num_prompts: int) -> List[str]:
    """Prepare prompts based on dataset type"""
    if dataset_type == "code":
        dataset = load_dataset("google-research-datasets/mbpp", "full")
        prompts = dataset["test"]["text"]
    elif dataset_type == "summary":
        dataset = load_dataset("EdinburghNLP/xsum", split="train")
        system_prompt = "Make a summary of the following text:\n\n"
        prompts = [system_prompt + doc for doc in dataset["document"]]
    elif dataset_type == "chat":
        dataset = load_dataset("shibing624/sharegpt_gpt4", split="train")
        prompts = [conv[0]["value"] for conv in dataset["conversations"]]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if num_prompts == -1:
        return prompts
    return prompts[:num_prompts]


def cleanup_vllm(llm: LLM):
    """Clean up vLLM resources"""
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    logger.info("Successfully delete the llm pipeline and free the GPU memory.")


def run_offline_vllm(
    server_args: Dict, dataset_type: str, num_prompts: int, output_dir: Path
) -> SDMetrics:
    """Run vLLM with given configuration and measure performance"""
    logger.info(f"Initializing vLLM with config: {server_args}")

    llm = LLM(**server_args)
    sampling_params = SamplingParams(temperature=0, max_tokens=256)

    prompts = prepare_prompts(dataset_type, num_prompts)
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

    outputs = []

    start = time.time()
    for message in tqdm(messages, desc="Generating outputs"):
        output = llm.chat(message, sampling_params)
        outputs.append(output[0])
    end = time.time()

    time_taken = end - start

    spec_config = server_args.get("speculative_config")
    mean_acceptance_length = None
    acceptance_rates = None

    if spec_config:
        num_spec_tokens = spec_config["num_speculative_tokens"]
        acceptance_counts = [0] * (num_spec_tokens + 1)

        for output in outputs:
            for step, count in enumerate(output.metrics.spec_token_acceptance_counts):
                acceptance_counts[step] += count

        mean_acceptance_length = sum(acceptance_counts) / acceptance_counts[0]
        acceptance_rates = [count / acceptance_counts[0] for count in acceptance_counts]

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    metrics = SDMetrics(
        main_model=server_args["model"],
        speculative_model=spec_config["model"] if spec_config else None,
        dataset_type=dataset_type,
        num_prompts=len(prompts),
        time_taken=time_taken,
        mean_acceptance_length=mean_acceptance_length,
        acceptance_rates=acceptance_rates,
        timestamp=timestamp.replace("_", " "),
    )

    output_file = output_dir / f"sd_results_{timestamp}.json"
    metrics.save_to_json(output_file)
    logger.info(f"Results saved to {output_file}")
    cleanup_vllm(llm)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run speculative decoding experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["code", "summary", "chat"],
        required=True,
        help="Dataset type to use",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=-1,
        help="Number of prompts to use (-1 for all)",
    )
    parser.add_argument(
        "--setup_type",
        type=str,
        choices=["single_setup", "few_setups"],
        required=True,
        help="Which setup to use from config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sd_experiments",
        help="Directory to save results",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.setup_type == "single_setup":
        run_offline_vllm(
            server_args=config["single_setup"]["server_args"],
            dataset_type=args.dataset,
            num_prompts=args.num_prompts,
            output_dir=output_dir,
        )
        main_model = config["single_setup"]["server_args"]["model"]
        speculative_model = (
            config["single_setup"]["server_args"]
            .get("speculative_config", {})
            .get("model", "None")
        )
        logger.info(
            f"Single setup completed: {main_model} {'with ' + speculative_model}"
        )

    else:
        for setup in tqdm(config["few_setups"]):
            main_model = setup["server_args"]["model"]
            speculative_model = (
                setup["server_args"].get("speculative_config", {}).get("model", "None")
            )
            try:
                run_offline_vllm(
                    server_args=setup["server_args"],
                    dataset_type=args.dataset,
                    num_prompts=args.num_prompts,
                    output_dir=output_dir,
                )
                logger.info(
                    f"Setup completed: {main_model} {'with ' + speculative_model}"
                )
            except Exception as e:
                error_msg = (
                    f"Setup failed: {main_model} {'with ' + speculative_model}:\n"
                    f"{str(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                logger.error(error_msg)
                continue


if __name__ == "__main__":
    main()
