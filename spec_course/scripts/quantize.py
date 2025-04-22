import argparse
import gc
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from datasets import load_dataset
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import LOG_PATH, load_config, setup_logger

CALIBRATION_DATASET = "neuralmagic/LLM_compression_calibration"


def load_calibration_dataset(
    tokenizer: AutoTokenizer, num_calibration_samples: int = 512
):
    def preprocess_fn(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
        }

    ds = load_dataset(CALIBRATION_DATASET, split="train")
    ds = ds.shuffle().select(range(num_calibration_samples))
    ds = ds.map(preprocess_fn)
    return ds


function_map = {
    "GPTQModifier": GPTQModifier,
}


def quantize_model(
    model_name,
    output_dir,
    logger,
    quant_method: str,
    quant_args: Dict[str, Any],
    num_calibration_samples: int = 512,
    max_sequence_length: int = 8192,
):
    logger.info(f"Processing model: {model_name}")
    iteration_parameters = quant_args.get("iteration_parameters")
    print("HERE", iteration_parameters)
    for key in iteration_parameters["keys"]:
        for values in iteration_parameters["values_per_keys"]:
            for value in values:
                model_output_dir = (
                    Path(output_dir) / f"{model_name.split('/')[-1]}-{key}-{value}"
                )
                model_output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    logger.info("Loading model and tokenizer...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, torch_dtype="auto", device_map="auto"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_name)

                    logger.info("Preparing calibration dataset...")
                    ds = load_calibration_dataset(tokenizer, num_calibration_samples)

                    logger.info("Preparing quantization recipe...")
                    static_parameters = quant_args.get("static_parameters", {})
                    recipe = function_map[quant_method](
                        **{**static_parameters, **{key: value}}
                    )

                    logger.info("Starting quantization...")
                    oneshot(
                        model=model,
                        output_dir=str(model_output_dir),
                        dataset=ds,
                        recipe=recipe,
                        max_seq_length=max_sequence_length,
                        num_calibration_samples=num_calibration_samples,
                        save_compressed=True,
                    )
                    model.save_pretrained(str(model_output_dir), save_compressed=True)
                    tokenizer.save_pretrained(str(model_output_dir))
                    logger.info(f"Model quantized and saved to: {model_output_dir}")
                except Exception as e:
                    logger.error(
                        f"Error processing model {model_name} with {key} {value}: {e}"
                    )
                    if "model" in locals():
                        del model
                    if "tokenizer" in locals():
                        del tokenizer
                    if "ds" in locals():
                        del ds

                    torch.cuda.empty_cache()
                    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Quantize language models to 8-bit")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", type=str, help="Single model to quantize")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory for quantized models",
    )
    parser.add_argument(
        "--num-samples", type=int, default=512, help="Number of calibration samples"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=8192, help="Maximum sequence length"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(LOG_PATH, "quantization")

    if args.config:
        config = load_config(args.config)
        models = config["models"]
        num_samples = config.get("num_calibration_samples", 512)
        max_seq_length = config.get("max_sequence_length", 8192)
        quant_methods = config.get("quant_methods", None)
    else:
        if not args.model:
            logger.error("Either --config or --model must be specified")
            return
        models = [args.model]
        num_samples = args.num_samples
        max_seq_length = args.max_seq_length

    if not quant_methods:
        raise ValueError("The quantization schema is not specified")

    for model_name in models:
        for quant_method, quant_args in quant_methods.items():
            try:
                quantize_model(
                    model_name=model_name,
                    output_dir=args.output_dir,
                    logger=logger,
                    quant_method=quant_method,
                    quant_args=quant_args,
                    num_calibration_samples=num_samples,
                    max_sequence_length=max_seq_length,
                )
            except Exception as e:
                logger.error(
                    f"Error processing model {model_name} with method {quant_method}: {e}"
                )
                continue


if __name__ == "__main__":
    main()
