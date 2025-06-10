from pathlib import Path
from typing import Any, Dict, Tuple


def parse_model_name(full_name: str) -> Tuple[str, str]:
    """Parse the model name and quantization type from a full model name."""
    model_name = full_name.split("/")[-1]

    if "scheme" in model_name:
        quantization_type = model_name.split("scheme")[-1].replace("-", "")
        model_name = model_name.split("scheme")[0][:-1]
    else:
        quantization_type = "FP16"

    return model_name, quantization_type


class ETLBase:
    def __init__(self, db_name: str):
        self.db_name = db_name

    def _extract(self, file_path: Path | str) -> Any:
        """
        Extract data from the source.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _transform(self, data: Any) -> Dict[Any, Any]:
        """
        Transform the extracted data.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _load(self, data: Dict[Any, Any]) -> None:
        """
        Load the transformed data into the target.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def run(self, file_path: Path | str) -> None:
        """
        Run the ETL process.
        """
        data = self._extract(file_path)
        transformed_data = self._transform(data)
        self._load(transformed_data)
