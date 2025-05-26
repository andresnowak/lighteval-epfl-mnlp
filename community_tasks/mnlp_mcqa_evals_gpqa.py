from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
import random

def mmlu_harness(line, task_name: str = None):
    topic = "knowledge and kills in advanced master-level STEM courses"
    prompt = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}. I will give you a mulitple choice question with its options, your answer should only be the letter of the option you choose (e.g A is the answer)\n"
    prompt += line["Question"] + "\n"
    options = [line["Correct Answer"], line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    gold_ix = random.randint(0, 3)
    options[0], options[gold_ix] = options[gold_ix], options[0]
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, options)])
    prompt += "Answer:"

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        instruction=f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}. I will give you a mulitple choice question with its options, your answer should only be the letter of the option you choose (e.g A is the answer)\n",
    )

task_gpqa = LightevalTaskConfig(
    name="mnlp_mcqa_evals_gpqa",
    prompt_function=mmlu_harness,
    suite=["community"],
    hf_subset="gpqa_main",
    hf_repo="Idavidrein/gpqa",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)