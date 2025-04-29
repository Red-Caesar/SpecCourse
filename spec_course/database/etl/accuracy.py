import time
from pathlib import Path
from typing import Any, Dict

from spec_course.database.db import (
    get_model_id,
    get_quantization_id,
    insert_accuracy,
    insert_model,
    insert_quantization,
)
from spec_course.database.etl.base import ETLBase


class Accuracy(ETLBase):
    def __init__(self, db_name: str) -> None:
        super().__init__(db_name)

    def _extract(self, file_path: Path | str) -> Any:
        with open(file_path, "r") as f:
            data = f.read()
        return data

    def _transform(self, data: Any) -> Dict[Any, Any]:
        import json

        data = json.loads(data)

        results = data["results"]["gsm8k"]
        full_model_name = data["configs"]["gsm8k"]["metadata"]["pretrained"]
        model_name = full_model_name.split("/")[-1]

        if "scheme" in model_name:
            quantization_type = model_name.split("scheme")[-1].replace("-", "")
            model_name = model_name.split("scheme")[0][:-1]
        else:
            quantization_type = "FP16"

        timestamp = data["date"]
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(timestamp))

        transformed_data = {
            "accuracy": results["exact_match,flexible-extract"],
            "model_name": model_name,
            "quantization_type": quantization_type,
            "date": date,
        }
        return transformed_data

    def _load(self, data: Dict[Any, Any]) -> None:
        model_id = get_model_id(self.db_name, data["model_name"])
        if model_id is None:
            model_id = insert_model(self.db_name, data["model_name"])

        quantization_id = get_quantization_id(self.db_name, data["quantization_type"])
        if quantization_id is None:
            quantization_id = insert_quantization(
                self.db_name, data["quantization_type"]
            )

        insert_accuracy(
            self.db_name, model_id, quantization_id, data["accuracy"], data["date"]
        )
