import argparse
import gc
import traceback
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_dataset
from llmcompressor.modifiers.obcq import SparseGPTModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import load_config, setup_logger

CALIBRATION_DATASET = "neuralmagic/LLM_compression_calibration"
logger = setup_logger(log_name="quantization")


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


str2QuantModifier = {
    "GPTQModifier": GPTQModifier,
    "SmoothQuantModifier": SmoothQuantModifier,
    "SparseGPTModifier": SparseGPTModifier,
}


def quantize_model(
    model_name,
    output_dir,
    quant_method: Dict[str, Any],
    num_calibration_samples: int = 512,
    max_sequence_length: int = 8192,
):
    logger.info(f"Processing model: {model_name}")
    model_output_dir = (
        Path(output_dir)
        / f"{model_name.split('/')[-1]}-scheme-{quant_method['model_suffix']}"
    )
    model_output_dir.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Preparing calibration dataset...")
        calibration_dataset = load_calibration_dataset(
            tokenizer, num_calibration_samples
        )

        logger.info("Preparing quantization recipe...")
        recipe = []
        for method in quant_method["recipe"]:
            method_name, method_args = list(method.items())[0]
            recipe.append(str2QuantModifier[method_name](**method_args))

        logger.info("Starting quantization...")
        oneshot(
            model=model,
            output_dir=str(model_output_dir),
            dataset=calibration_dataset,
            recipe=recipe,
            max_seq_length=max_sequence_length,
            num_calibration_samples=num_calibration_samples,
            save_compressed=True,
            oneshot_device="auto",
        )
        logger.info(f"Model quantized and saved to: {model_output_dir}")
    except Exception as e:
        error_msg = (
            f"Error processing model {model_name} with method {quant_method['setup_name']}:\n"
            f"{str(e)}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        logger.error(error_msg)
        if "model" in locals():
            del model
        if "tokenizer" in locals():
            del tokenizer
        if "calibration_dataset" in locals():
            del calibration_dataset

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
        for quant_method in quant_methods:
            quantize_model(
                model_name=model_name,
                output_dir=args.output_dir,
                quant_method=quant_method,
                num_calibration_samples=num_samples,
                max_sequence_length=max_seq_length,
            )


if __name__ == "__main__":
    main()
