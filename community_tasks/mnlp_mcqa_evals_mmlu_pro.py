from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig

def mmlu_pro_extractor(line, task_name: str = None) -> tuple[str, list[str], str, int]:
    topic = "knowledge and kills in advanced master-level STEM courses"
    prompt = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"] + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["options"])])
    prompt += "Answer:"
    gold_ix = LETTER_INDICES.index(line["answer"])

    return line["question"], line["options"], line["answer"], gold_ix
