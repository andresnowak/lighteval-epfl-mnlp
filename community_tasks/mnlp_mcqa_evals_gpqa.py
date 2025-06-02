from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
import random

def gpqa_extractor(line, task_name: str = None) -> tuple[str, list[str], str, int]:
    options = [
    ("A", line["Correct Answer"]),
    ("B", line["Incorrect Answer 1"]),
    ("C", line["Incorrect Answer 2"]),
    ("D", line["Incorrect Answer 3"]),
    ]

    # Shuffle the options to prevent positional bias
    random.shuffle(options)

    # Extract the text of each option after shuffling
    choices = [f" {label}. {text}" for label, text in options]

    # Determine the index of the correct answer *after* shuffling
    gold_ix = next(i for i, (_, text) in enumerate(options) if text == line["Correct Answer"])

    return line["Question"], choices, line["Correct Answer"], gold_ix
