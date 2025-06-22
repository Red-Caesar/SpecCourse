import json
from pathlib import Path
from typing import Any, Dict

from spec_course.database.db import (
    get_dataset_id,
    get_model_id,
    get_quantization_id,
    get_sd_setup_id,
    insert_dataset,
    insert_load_test_performance,
    insert_model,
    insert_quantization,
    insert_sd_setup,
)
from spec_course.database.etl.base import ETLBase


class LoadTestETL(ETLBase):
    def __init__(self, db_name: str) -> None:
        super().__init__(db_name)

    def _extract(self, folder_path: Path | str) -> Any:
        folder = Path(folder_path)
        with open(folder / "input_params.json", "r") as f:
            input_params = json.load(f)
        with open(folder / "metrics.json", "r") as f:
            metrics = json.load(f)
        return {
            "input_params": input_params,
            "metrics": metrics,
            "folder_name": folder.name,
        }

    def _transform(self, data: Any) -> Dict[Any, Any]:
        input_params = data["input_params"]
        metrics = data["metrics"]

        rps = int(input_params.get("rps", "1"))

        run_id = input_params.get("run_id", "")
        num_spec_tokens = 0
        if "sd_" in run_id:
            try:
                num_spec_tokens = int(run_id.split("sd_")[1])
            except Exception:
                num_spec_tokens = 0

        latency = (
            metrics.get("metrics", {}).get("end_to_end_latency", {}).get("med", 0.0)
        )

        # TODO: Think how to do it better
        target_model_name, target_quantization = (
            "meta-llama/Llama-3.1-8B-Instruct",
            "FP16",
        )
        if "single_model" not in run_id:
            draft_model_name = "Llama-3.2-1B-Instruct"
            draft_quantization = "FP8"
        else:
            draft_model_name = ""
            draft_quantization = ""

        dataset_type = "code"
        date = "_".join(data["folder_name"].split("_")[-2:])

        transformed = {
            "target_model": target_model_name,
            "target_quantization": target_quantization,
            "draft_model": draft_model_name,
            "draft_quantization": draft_quantization,
            "dataset_type": dataset_type,
            "rps": rps,
            "end_to_end_latency": latency,
            "num_spec_tokens": num_spec_tokens,
            "date": date,
        }
        return transformed

    def _load(self, data: Dict[Any, Any]) -> None:
        target_model_id = get_model_id(self.db_name, data["target_model"])
        if target_model_id is None:
            target_model_id = insert_model(self.db_name, data["target_model"])

        draft_model_id = get_model_id(self.db_name, data["draft_model"])
        if draft_model_id is None:
            draft_model_id = insert_model(self.db_name, data["draft_model"])

        target_quantization_id = get_quantization_id(
            self.db_name, data["target_quantization"]
        )
        if target_quantization_id is None:
            target_quantization_id = insert_quantization(
                self.db_name, data["target_quantization"]
            )

        draft_quantization_id = get_quantization_id(
            self.db_name, data["draft_quantization"]
        )
        if draft_quantization_id is None:
            draft_quantization_id = insert_quantization(
                self.db_name, data["draft_quantization"]
            )

        dataset_id = get_dataset_id(self.db_name, data["dataset_type"])
        if dataset_id is None:
            dataset_id = insert_dataset(self.db_name, data["dataset_type"])

        sd_setup_id = get_sd_setup_id(
            self.db_name,
            target_model_id,
            target_quantization_id,
            draft_model_id,
            draft_quantization_id,
            dataset_id,
        )
        if sd_setup_id is None:
            sd_setup_id = insert_sd_setup(
                self.db_name,
                target_model_id,
                target_quantization_id,
                draft_model_id,
                draft_quantization_id,
                dataset_id,
            )

        insert_load_test_performance(
            self.db_name,
            sd_setup_id,
            data["rps"],
            data["end_to_end_latency"],
            data["num_spec_tokens"],
            data["date"],
        )
