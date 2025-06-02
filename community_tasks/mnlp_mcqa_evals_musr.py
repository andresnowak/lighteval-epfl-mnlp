from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
import ast

def musr_extractor(line, task_name: str = None) -> tuple[str, str, list[str], str, int]:
    choices = ast.literal_eval(line["choices"])
    gold_ix = line["answer_index"]
 
    return line['question'], line["narrative"], choices, line["answer_index"], gold_ix
