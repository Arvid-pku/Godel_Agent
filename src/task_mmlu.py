import random
import string
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas
import json
from openai import OpenAI
import numpy as np
import time
from wrap import wrap_solver

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])
client = OpenAI()
threshold = 0.80
last_test_acc = 0.

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question.
Question's Subject: {Subject}
Question: {Question}

Choices:
(A) {A}
(B) {B}
(C) {C}
(D) {D}
""".strip()

def solver(agent, task: str):
    messages = [{"role": "user", "content": f"# Your Task:\n{task}"}]
    response = agent.action_call_json_format_llm(
        model="gpt-3.5-turbo", 
        messages=messages, 
        temperature=0.8, 
        num_of_response=1,
        role="knowledge and reasoning expert", 
        return_dict_keys=["reasoning", "answer"], 
        requirements=(
            "1. Please explain step by step.\n"
            "2. The answer MUST be either A or B or C or D.\n"
        ).strip(), 
    )
    
    return_dict = response[0]
    return_dict["answer"] = str(return_dict.get("answer", ""))
    return return_dict

def real_evaluate(solver):
    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    # set seed 0 for valid set
    data_filename = "../dataset/mmlu.csv"
    df = pandas.read_csv(data_filename)
    random.seed(0)
    examples = [row.to_dict() for _, row in df.iterrows()]
    random.shuffle(examples)
    examples = examples[128:928]
    questions = [format_multichoice_question(example) for example in examples]
    answers = [example['Answer'] for example in examples]

    max_workers = min(len(examples), 48)
    task_queue = []
    for q in questions:
        task_queue.append(q)
    acc_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(wrap_solver(solver), task_queue), total=len(task_queue)))
    info_list = []
    for q_idx, res in enumerate(results):
        try:
            extracted_answer = str(res["answer"])
            for a in ["A", "B", "C", "D"]:
                if (a + ")") in extracted_answer or f"'{a}'" in extracted_answer:
                    extracted_answer = a
            correct_answer = str(answers[q_idx])
        except Exception as e:
            info_list.append(f"Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
            acc_list.append(0)
            continue
        acc_list.append(extracted_answer == correct_answer)
        info_list.append(f"Sample {q_idx}:\n{task_queue[q_idx]}\nModel Output: {res}\nModel Answer: {extracted_answer}\nCorrect Answer: {correct_answer}\nIs Correct: {acc_list[-1]}\n")

    acc = sum(acc_list) / len(acc_list)
    interval = bootstrap_confidence_interval(acc_list)
    if acc > last_test_acc:
        open(f"result/mmlu_{round(acc, 4)}.txt", "w").writelines([interval] + info_list)
    return acc

class MMLU_Task:
    def evaluate(self, solver):
        # LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        # set seed 0 for valid set
        data_filename = "../dataset/mmlu.csv"
        df = pandas.read_csv(data_filename)
        random.seed(0)
        examples = [row.to_dict() for _, row in df.iterrows()]
        random.shuffle(examples)
        examples = examples[:128]
        random.seed(time.time())
        random.shuffle(examples)
        examples = examples[:20]
        questions = [format_multichoice_question(example) for example in examples]
        answers = [example['Answer'] for example in examples]

        max_workers = min(len(examples), 48)
        task_queue = []
        for q in questions:
            task_queue.append(q)
        acc_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(solver, task_queue), total=len(task_queue)))
        info_list = []
        for q_idx, res in enumerate(results):
            try:
                extracted_answer = str(res["answer"])
                for a in ["A", "B", "C", "D"]:
                    if (a + ")") in extracted_answer or f"'{a}'" in extracted_answer:
                        extracted_answer = a
                correct_answer = str(answers[q_idx])
            except Exception as e:
                info_list.append(f"Valid Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
                acc_list.append(0)
                continue
            acc_list.append(extracted_answer == correct_answer)
            info_list.append(f"Valid Sample {q_idx}:\n{task_queue[q_idx]}\nModel Output: {res}\nModel Answer: {extracted_answer}\nCorrect Answer: {correct_answer}\nIs Correct: {acc_list[-1]}\n")

        valid_acc = sum(acc_list) / len(acc_list)
        print("Acc:", valid_acc)
        if valid_acc >= threshold:
            test_acc = real_evaluate(solver)
            feedback = f"Valid Accuracy: {valid_acc}\nTest Accuracy {test_acc}\n" + "Evaluation Info:\n" + "\n".join(info_list)
        else:
            test_acc = 0
            feedback = f"Valid Accuracy: {valid_acc}\nValid Accuracy less than {threshold}, no testing needed.\n" + "Evaluation Info:\n" + "\n".join(info_list)
        return feedback, test_acc

def format_multichoice_question(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)



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
    print(f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%")
    # Return the formatted string with confidence interval and median
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"


# real_evaluate(original_solver)