from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from .mnlp_mcqa_evals_mmlu import task_mmlu
from .mnlp_mcqa_evals_mmlu_reasoning import task_mmlu_reasoning
from .mnlp_mcqa_evals_mmlu_pro import task_mmlu_pro
from .mnlp_mcqa_evals_arc_easy import task_arc_easy
from .mnlp_mcqa_evals_arc_challenge import task_arc_challenge
from .mnlp_mcqa_evals_gpqa import task_gpqa
from .mnlp_mcqa_evals_musr import task_musr
from .mnlp_mcqa_evals_nlp4education import task_nlp4education
from .mnlp_mcqa_evals_math_qa import task_math_qa

def mmlu_harness(line, task_name: str = None):
    topic = "knowledge and kills in advanced master-level STEM courses"
    prompt = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"] + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    prompt += "Answer:"
    gold_ix = LETTER_INDICES.index(line["answer"])

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        instruction=f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n",
    )

task_example = LightevalTaskConfig(
    name="mnlp_mcqa_evals",
    prompt_function=mmlu_harness,
    suite=["community"],
    hf_subset="",
    hf_repo="zechen-nlp/MNLP_STEM_mcqa_demo",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

# STORE YOUR EVALS
TASKS_TABLE = [task_example, task_mmlu, task_mmlu_pro, task_arc_easy, task_arc_challenge, task_gpqa, task_musr, task_nlp4education, task_mmlu_reasoning, task_math_qa]