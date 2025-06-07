from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from .mnlp_mcqa_evals_mmlu import mmlu_extractor
from .mnlp_mcqa_evals_mmlu_reasoning import task_mmlu_reasoning
from .mnlp_mcqa_evals_mmlu_pro import mmlu_pro_extractor
from .mnlp_mcqa_evals_arc_easy import arc_easy_extractor
from .mnlp_mcqa_evals_arc_challenge import arch_challenge_extractor
from .mnlp_mcqa_evals_gpqa import gpqa_extractor
from .mnlp_mcqa_evals_musr import musr_extractor
from .mnlp_mcqa_evals_nlp4education import nlp4education_extractor
from .mnlp_mcqa_evals_math_qa import math_qa_extractor
from .mnlp_mcqa_evals_mcqa_evals import mcqa_evals_extractor

import random

random.seed(42)


def prompt_creator_musr(question, choices, narrative: str, prompt_type: int=0):
    topic = "knowledge and kills in advanced master-level STEM courses"
    instruction = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    prompt = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    prompt += f"Narrative: {narrative}" + "\n"
    prompt += f"Question: {question}" + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, choices)])
    prompt += "Answer:"

    return prompt, topic, instruction


def prompt_creator(question, choices, prompt_type: int=0):
    choices = "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, choices)])

    if prompt_type == 0:
        topic = "knowledge and skills in advanced master-level STEM courses"
        instruction = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"

        prompt = f"""{instruction}{question}\n{choices}\nAnswer:"""
    elif prompt_type == 1:
        topic = "knowledge and skills in advanced master-level STEM courses"
        instruction = (
            f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
        )

        prompt = f"""{instruction}{question}\n{choices}\nFor your answer you just need to select the option you think is correct and output in this format of letter. text answer (e.g A. water or B. The human body has 206 bones), nothing else. \nAnswer:"""

    elif prompt_type == 2:
        topic = "graduate-level science, technology, engineering, and mathematics (STEM) concepts"
        instruction = (
            f"This is part of an assessment on {topic.replace('_', ' ')}. Each question is multiple-choice and requires a single correct answer.\n\n"
        )

        prompt = f"""{instruction}{question}\n{choices}\nFor grading purposes, respond with: [LETTER]. [VERBATIM TEXT]\nExample: D. Planck constant\nYour Response:"""
    elif prompt_type == 3:
        topic = "complex STEM concepts typically taught in advanced university courses"
        instruction = (
            f"You're a helpful tutor reviewing {topic.replace('_', ' ')}. The following is a multiple-choice question with several options. Select the best answer.\n\n"
        )

        prompt = f"""{instruction}{question}\n{choices}\nGive only the letter and the complete answer text. No explanation needed.\nAnswer:"""
    elif prompt_type == 4:
        topic = "STEM problem-solving at the postgraduate level"
        instruction = (
            f"Choose the best answer from the options below. Each question tests {topic.replace('_', ' ')}.\n\n"
        )

        prompt = f"""{instruction}{question}\n{choices}\nRespond with the correct letter and answer (e.g., D. Kirchhoff's voltage law).\nYour final answer is:"""
    elif prompt_type == 5:
        topic = "challenging STEM problems as found on graduate standardized tests"
        instruction = (
            f"This question assesses {topic.replace('_', ' ')}. Carefully evaluate the options and select the correct answer.\n\n"
        )

        prompt = f"""{instruction}{question}\n{choices}\nYour response should include the letter and the exact text of the correct choice. Example: B. Entropy increases.\nAnswer:"""

    return prompt, topic, instruction


# Types of questions
def single_letter(line, task_name: str = None):
    if task_name == "community|mnlp_mcqa_evals_arc_easy":
        question, choices, answer, gold_idx = arc_easy_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_mcqa_evals":
        question, choices, answer, gold_idx = mcqa_evals_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_arc_challenge":
        question, choices, answer, gold_idx = arch_challenge_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_gpqa":
        question, choices, answer, gold_idx = gpqa_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_math_qa":
        question, choices, answer, gold_idx = math_qa_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_mmlu_pro":
        question, choices, answer, gold_idx = mmlu_pro_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_mmlu":
        question, choices, answer, gold_idx = mmlu_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_nlp4education":
        question, choices, answer, gold_idx = nlp4education_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_musr":
        question, narrative, choices, answer, gold_idx = musr_extractor(line)
    else:
        raise ValueError(
            f"Unsupported task_name: '{task_name}'."
        )

    if task_name == "community|mnlp_mcqa_evals_musr":
        prompt, topic, instruction = prompt_creator_musr(question, choices, narrative)
    else:
        prompt, topic, instruction = prompt_creator(question, choices, prompt_type=0)

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {chr(65 + i)}" for i in range(len(choices))],
        gold_index=gold_idx,
        instruction=instruction,
    )


def letter_answer(line, task_name: str):
    if task_name == "community|mnlp_mcqa_evals_arc_easy":
        question, choices, answer, gold_idx = arc_easy_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_mcqa_evals":
        question, choices, answer, gold_idx = mcqa_evals_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_arc_challenge":
        question, choices, answer, gold_idx = arch_challenge_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_gpqa":
        question, choices, answer, gold_idx = gpqa_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_math_qa":
        question, choices, answer, gold_idx = math_qa_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_mmlu_pro":
        question, choices, answer, gold_idx = mmlu_pro_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_mmlu":
        question, choices, answer, gold_idx = mmlu_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_nlp4education":
        question, choices, answer, gold_idx = nlp4education_extractor(line)
    elif task_name == "community|mnlp_mcqa_evals_musr":
        question, narrative, choices, answer, gold_idx = musr_extractor(line)
    else:
        raise ValueError(
            f"Unsupported task_name: '{task_name}'."
        )

    if task_name == "community|mnlp_mcqa_evals_musr":
        prompt, topic, instruction = prompt_creator_musr(question, choices, narrative)
    else:
        prompt, topic, instruction = prompt_creator(question, choices, prompt_type=0)

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {chr(65 + i)}. {choices[i]}" for i in range(len(choices))],
        gold_index=gold_idx,
        instruction=instruction,
    )

# Define the tasks

prompt_type = single_letter

task_arc_easy = LightevalTaskConfig(
    name="mnlp_mcqa_evals_arc_easy",
    prompt_function=prompt_type,
    suite=["community"],
    hf_subset="ARC-Easy",
    hf_repo="allenai/ai2_arc",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["validation", "test"],
    evaluation_splits=["validation", "test"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

task_mcqa_evals = LightevalTaskConfig(
    name="mnlp_mcqa_evals_mcqa_evals",
    prompt_function=prompt_type,
    suite=["community"],
    hf_subset="",
    hf_repo="zechen-nlp/MNLP_STEM_mcqa_evals",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

task_arc_challenge = LightevalTaskConfig(
    name="mnlp_mcqa_evals_arc_challenge",
    prompt_function=prompt_type,
    suite=["community"],
    hf_subset="ARC-Challenge",
    hf_repo="allenai/ai2_arc",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["validation", "test"],
    evaluation_splits=["validation", "test"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

task_gpqa = LightevalTaskConfig(
    name="mnlp_mcqa_evals_gpqa",
    prompt_function=prompt_type,
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

task_math_qa = LightevalTaskConfig(
    name="mnlp_mcqa_evals_math_qa",
    prompt_function=prompt_type,
    suite=["community"],
    hf_subset="math_qa",
    hf_repo="andresnowak/MNLP_MCQA_dataset",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["validation"],
    evaluation_splits=["validation"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

task_mmlu_pro = LightevalTaskConfig(
    name="mnlp_mcqa_evals_mmlu_pro",
    prompt_function=prompt_type,
    suite=["community"],
    hf_subset="",
    hf_repo="igzi/mmlu-pro-stem",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["validation", "test"],
    evaluation_splits=["validation", "test"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

task_mmlu = LightevalTaskConfig(
    name="mnlp_mcqa_evals_mmlu",
    prompt_function=prompt_type,
    suite=["community"],
    hf_subset="",
    hf_repo="TIGER-Lab/MMLU-STEM",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test", "validation"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

task_nlp4education = LightevalTaskConfig(
    name="mnlp_mcqa_evals_nlp4education",
    prompt_function=prompt_type,
    suite=["community"],
    hf_subset="",
    hf_repo="igzi/nlp4education",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

task_musr = LightevalTaskConfig(
    name="mnlp_mcqa_evals_musr",
    prompt_function=prompt_type,
    suite=["community"],
    hf_subset="",
    hf_repo="TAUR-Lab/MuSR",  # Change the repo name to the evaluation dataset that you compiled
    hf_avail_splits=["murder_mysteries", "object_placements","team_allocation"],
    evaluation_splits=["murder_mysteries", "object_placements","team_allocation"],
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0,  # Set to 0 to use all samples, specify a number to limit the number of samples for debugging purposes
)

# ------------

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
TASKS_TABLE = [task_example, task_mmlu, task_mmlu_pro, task_arc_easy, task_arc_challenge, task_gpqa, task_musr, task_nlp4education, task_mmlu_reasoning, task_math_qa, task_mcqa_evals]