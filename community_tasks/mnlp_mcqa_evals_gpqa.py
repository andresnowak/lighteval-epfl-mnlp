from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
import random

def gpqa_extractor(line, task_name: str = None) -> tuple[str, list[str], str, int]:
    options = [line["Correct Answer"], line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]  # this method may not be good because we are always putting everything in the same order (so if the model has a bias it couold be very good here)
    gold_ix = random.randint(0, 3)
    options[0], options[gold_ix] = options[gold_ix], options[0]

    choices = [
        f" A. {options[0]}",
        f" B. {options[1]}",
        f" C. {options[2]}",
        f" D. {options[3]}",
    ]

    return line["Question"], choices, line["Correct Answer"], gold_ix
