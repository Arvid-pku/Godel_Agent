import random
import string
from collections import namedtuple
from typing import List
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import pandas as pd
from wrap import wrap_solver

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index', 'domain'])
client = OpenAI()
threshold = 0.31
last_test_acc = 0.

def solver(agent, task: str):
    messages = [{"role": "user", "content": f"# Your Task:\n{task}"}]
    response = agent.action_call_json_format_llm(
        model="gpt-3.5-turbo", 
        messages=messages, 
        temperature=0.8, 
        num_of_response=1,
        role="science professor", 
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
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    data_filename = '../dataset/gpqa_diamond.csv'
    INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    # set seed 0 for valid set
    questions = load_questions(data_filename, seed=0)
    val_questions = questions[32:]
    max_workers = min(len(val_questions), 48)

    task_queue = []
    for q in val_questions:
        task_content = f"Answer the following multiple choice question.\n" \
                    + f"Question's Domain: {q.domain}\nQuestion: {q.question}" \
                    + f"\n\nChoices:\n(A) {q.choice1}\n(B) {q.choice2}\n(C) {q.choice3}\n(D) {q.choice4}"
        task_queue.append(task_content)


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
            correct_answer = INDEX_TO_LETTER[val_questions[q_idx].correct_index]
        except Exception as e:
            info_list.append(f"Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
            acc_list.append(0)
            continue
        acc_list.append(extracted_answer == correct_answer)
        info_list.append(f"Sample {q_idx}:\n{task_queue[q_idx]}\nModel Output: {res}\nModel Answer: {extracted_answer}\nCorrect Answer: {correct_answer}\nIs Correct: {acc_list[-1]}\n")

    acc = sum(acc_list) / len(acc_list)
    interval = bootstrap_confidence_interval(acc_list)
    if acc > last_test_acc:
        open(f"result/gpqa_{round(acc, 4)}.txt", "w").writelines([interval] + info_list)
    return acc

class GPQA_Task:
    def evaluate(self, solver):
        # dynamically define forward()
        # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
        data_filename = '../dataset/gpqa_diamond.csv'
        INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        # set seed 0 for valid set
        questions = load_questions(data_filename, seed=0)
        val_questions = questions[:32]
        max_workers = min(len(val_questions), 48)

        task_queue = []
        for q in val_questions:
            task_content = f"Answer the following multiple choice question.\n" \
                    + f"Question's Domain: {q.domain}\nQuestion: {q.question}" \
                    + f"\n\nChoices:\n(A) {q.choice1}\n(B) {q.choice2}\n(C) {q.choice3}\n(D) {q.choice4}"
            task_queue.append(task_content)

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
                correct_answer = INDEX_TO_LETTER[val_questions[q_idx].correct_index]
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


def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


def load_questions(path: str, seed: int) -> List[Example]:
    """Load questions from csv file and return a list of Example namedtuples."""
    question_df = pd.read_csv(path)
    random.seed(seed)

    def shuffle_choices_and_create_example(row) -> Example:
        list_choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
        random.shuffle(list_choices)
        example = Example(row.Question, list_choices[0], list_choices[1], list_choices[2], list_choices[3],
                          list_choices.index(row['Correct Answer']), row["High-level domain"])
        return example

    return [shuffle_choices_and_create_example(row) for _, row in question_df.iterrows()]


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