"""Dataset background prompts used by the Time-MaC text branch."""

from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROMPT_FILES = {
    "etth1": "ETTh1.txt",
    "etth2": "ETTh2.txt",
    "ettm1": "ETTm1.txt",
    "ettm2": "ETTm2.txt",
    "electricity": "Electricity.txt",
    "ecl": "Electricity.txt",
    "weather": "Weather.txt",
    "traffic": "Traffic.txt",
}


def resolve_prompt_bank_dir(prompt_bank_dir: str) -> Path:
    """Resolve a prompt-bank directory independently of the current directory."""
    path = Path(prompt_bank_dir).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_dataset_prompt(dataset: str, prompt_bank_dir: str = "./prompt_bank") -> Tuple[str, Path]:
    """Load the tracked background description for a supported benchmark dataset."""
    key = dataset.strip().lower()
    if key not in PROMPT_FILES:
        supported = ", ".join(sorted(PROMPT_FILES))
        raise ValueError(f"No prompt-bank mapping for dataset '{dataset}'. Supported names: {supported}")

    prompt_path = resolve_prompt_bank_dir(prompt_bank_dir) / PROMPT_FILES[key]
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Prompt-bank file not found for dataset '{dataset}': {prompt_path}"
        )

    content = prompt_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt-bank file is empty: {prompt_path}")
    return content, prompt_path
