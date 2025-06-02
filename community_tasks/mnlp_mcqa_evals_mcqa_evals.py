from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig


def mcqa_evals_extractor(line, task_name: str = None) -> tuple[str, list[str], str, int]:
    gold_ix = LETTER_INDICES.index(line["answer"])

    return line["question"], line["choices"], line["answer"], gold_ix
