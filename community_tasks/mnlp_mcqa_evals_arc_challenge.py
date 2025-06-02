from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig

def arch_challenge_extractor(line, task_name: str = None) -> tuple[str, list[str], str, int]:
    answer = line["answerKey"]
    if 'A' <= answer[0] <= 'Z':
        gold_ix = LETTER_INDICES.index(line["answerKey"])
    else:
        gold_ix = int(answer) - 1

    return line["question"], line["choices"]["text"], line["answerKey"], gold_ix
