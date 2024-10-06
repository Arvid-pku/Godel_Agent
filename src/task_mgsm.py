import json
import random
import time
import string
from tqdm import tqdm
from openai import OpenAI
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from wrap import wrap_solver

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])
threshold = 0.5
last_test_acc = 0.

client = OpenAI()
LANG_TO_INSTRUCTIONS = {
    "en": """Solve this math problem.

{input}""",
    "bn": """এই গণিতের সমস্যাটি সমাধান করুন।

{input}""",
    "de": """Löse dieses Mathematikproblem.

{input}""",
    "es": """Resuelve este problema matemático.

{input}""",
    "fr": """Résolvez ce problème de mathématiques.

{input}""",
    "ja": """この数学の問題を解いてください。

{input}""",
    "ru": """Решите эту математическую задачу.

{input}""",
    "sw": """Suluhisha tatizo hili la hesabu.

{input}""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి.

{input}""",
    "th": """แก้ปัญหาคณิตศาสตร์นี้

{input}""",
    "zh": """解决这个数学问题。

{input}"""
}

LANG_TO_FPATH = lambda lang: f"../dataset/mgsm/mgsm_{lang}.tsv"

ALL_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]

def solver(agent, task: str):
    messages = [{"role": "user", "content": f"# Your Task:\n{task}"}]
    response = agent.action_call_json_format_llm(
        model="gpt-3.5-turbo", 
        messages=messages, 
        temperature=0.8, 
        num_of_response=1,
        role="math expert", 
        return_dict_keys=["reasoning", "answer"], 
        requirements=(
            "1. Please explain step by step.\n"
            "2. The answer MUST be an integer.\n"
        ).strip(),
    )
    
    return_dict = response[0]
    return_dict["answer"] = str(return_dict.get("answer", ""))
    return return_dict

def real_evaluate(solver):
    examples = get_all_examples()
    random.seed(0)
    random.shuffle(examples)
    examples = examples[128:928]
    questions = [example['inputs'] for example in examples]
    answers = [example['targets'] for example in examples]
    max_workers = min(len(examples), 48)
    task_queue = []
    for q in questions:
        task_queue.append(q)

    acc_list = []
    info_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(wrap_solver(solver), task_queue), total=len(task_queue)))

    for q_idx, res in enumerate(results):
        try:
            extracted_answer = str(res["answer"])
            correct_answer = str(answers[q_idx])
            correct = score_mgsm(correct_answer, extracted_answer)
        except Exception as e:
            info_list.append(f"Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
            acc_list.append(0)
            continue
        acc_list.append(correct)
        info_list.append(f"Sample {q_idx}:\n{task_queue[q_idx]}\nModel Output: {res}\nModel Answer: {extracted_answer}\nCorrect Answer: {correct_answer}\nIs Correct: {acc_list[-1]}\n")

    acc = sum(acc_list) / len(acc_list)
    interval = bootstrap_confidence_interval(acc_list)
    if acc > last_test_acc:
        open(f"result/mgsm_{round(acc, 4)}.txt", "w").writelines([interval] + info_list)
    return acc

class MGSM_Task:
    def evaluate(self, solver):
        examples = get_all_examples()
        random.seed(0)
        random.shuffle(examples)
        examples = examples[:128]
        random.seed(time.time())
        random.shuffle(examples)
        examples = examples[:20]
        questions = [example['inputs'] for example in examples]
        answers = [example['targets'] for example in examples]
        max_workers = min(len(examples), 48)
        task_queue = []
        for q in questions:
            task_queue.append(q)
    
        acc_list = []
        info_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(solver, task_queue), total=len(task_queue)))

        for q_idx, res in enumerate(results):
            try:
                extracted_answer = str(res["answer"])
                correct_answer = str(answers[q_idx])
                correct = score_mgsm(correct_answer, extracted_answer)
            except Exception as e:
                info_list.append(f"Valid Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
                acc_list.append(0)
                continue
            acc_list.append(correct)
            info_list.append(f"Valid Sample {q_idx}:\n{task_queue[q_idx]}\nModel Output: {res}\nModel Answer: {extracted_answer}\nCorrect Answer: {correct_answer}\nIs Correct: {acc_list[-1]}\n")

        valid_acc = sum(acc_list) / len(acc_list)
        print("Acc:", valid_acc)
        if valid_acc >= threshold:
            test_acc = real_evaluate(solver)
            feedback = f"Valid Accuracy: {valid_acc}\nTest Accuracy {test_acc}\n" + "Evaluation Info:\n" + "\n".join(info_list)
        else:
            test_acc = 0
            feedback = f"`Valid Accuracy: `{valid_acc}\nValid Accuracy less than {threshold}, no testing needed.\n" + "Evaluation Info:\n" + "\n".join(info_list)
        return feedback, test_acc


def score_mgsm(target: str, prediction: str) -> bool:
    if "." in prediction:
        prediction = prediction.rstrip("0").rstrip(".")

    target = target.replace(",", "")
    prediction = prediction.replace(",", "")

    return target == prediction


def get_lang_examples(lang: str) -> list[dict[str, str]]:
    fpath = LANG_TO_FPATH(lang)
    examples = []
    with open(fpath, mode='r', encoding='utf-8') as f:
        for line in f:
            inputs, targets = line.strip().split("\t")
            if "." in targets:
                raise ValueError(f"targets {targets} contains a decimal point.")
            # targets = int(targets.replace(",", ""))
            examples.append({"inputs": LANG_TO_INSTRUCTIONS[lang].format(input=inputs), "targets": targets, "lang": lang})
    return examples


def get_all_examples() -> list[dict[str, str]]:
    examples = []
    for lang in ALL_LANGUAGES:
        # if lang != "en":
        #     continue
        examples += get_lang_examples(lang)
    return examples



def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.
    
    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    # Convert data to a numpy array for easier manipulation
    data = np.array(data)

    # List to store the means of bootstrap samples
    bootstrap_means = []

    # Generate bootstrap samples and compute the mean for each sample
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # Convert bootstrap_means to a numpy array for percentile calculation
    bootstrap_means = np.array(bootstrap_means)

    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # Compute the median of the bootstrap means
    median = np.median(bootstrap_means)

    # Convert to percentages and format to one decimal place
    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    # Return the formatted string with confidence interval and median
    print(f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%")
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"


# real_evaluate(original_solver)