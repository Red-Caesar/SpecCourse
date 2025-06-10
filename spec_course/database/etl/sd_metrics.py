from pathlib import Path
from typing import Any, Dict

from spec_course.database.db import (
    get_dataset_id,
    get_model_id,
    get_quantization_id,
    get_sd_setup_id,
    insert_dataset,
    insert_model,
    insert_quantization,
    insert_sd_performance,
    insert_sd_setup,
)
from spec_course.database.etl.base import ETLBase, parse_model_name


class SDMetrics(ETLBase):
    def __init__(self, db_name: str) -> None:
        super().__init__(db_name)

    def _extract(self, file_path: Path | str) -> Any:
        with open(file_path, "r") as f:
            data = f.read()
        return data

    def _transform(self, data: Any) -> Dict[Any, Any]:
        import json

        data = json.loads(data)

        target_model_name, target_quantization = parse_model_name(data["main_model"])
        if data["speculative_model"]:
            draft_model_name, draft_quantization = parse_model_name(
                data["speculative_model"]
            )
        else:
            draft_model_name = ""
            draft_quantization = ""

        date = data["timestamp"]

        mean_acceptance_length = (
            data["mean_acceptance_length"] if data["mean_acceptance_length"] else 0
        )
        acceptance_rates = (
            data["acceptance_rates"]
            if data["acceptance_rates"]
            else [0 for _ in range(5)]
        )
        assert len(acceptance_rates) == 5, "Acceptance rates must have 5 values."

        transformed_data = {
            "target_model": target_model_name,
            "target_quantization": target_quantization,
            "draft_model": draft_model_name,
            "draft_quantization": draft_quantization,
            "dataset_type": data["dataset_type"],
            "time_taken": data["time_taken"],
            "date": date,
            "mean_acceptance_length": mean_acceptance_length,
            "acceptance_rates": acceptance_rates,
        }
        return transformed_data

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

        insert_sd_performance(
            self.db_name,
            sd_setup_id,
            data["mean_acceptance_length"],
            data["date"],
            data["time_taken"],
            data["acceptance_rates"],
        )
