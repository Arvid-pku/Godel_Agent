# https://github.com/openai/simple-evals/blob/main/drop_eval.py
"""
DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs
Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, Matt Gardner
https://arxiv.org/abs/1903.00161
"""

import gzip
import json
import random
import re
import string
from typing import Any, Dict, List, Set, Tuple, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import linear_sum_assignment
from wrap import wrap_solver


threshold = 83
last_test_acc = 0.

def solver(agent, task: str):
    messages = [{"role": "user", "content": f"# Your Task:\n{task}"}]
    response = agent.action_call_json_format_llm(
        model="gpt-3.5-turbo", 
        messages=messages, 
        temperature=0.5, 
        num_of_response=1,
        role="read comprehension expert", 
        return_dict_keys=["reasoning", "answer"], 
        requirements=(
            "1. Please explain step by step.\n"
            "2. Directly answer the question.\n"
            "3. The answer MUST be a concise string.\n"
        ).strip(), 
    )
    
    return_dict = response[0]
    return_dict["answer"] = str(return_dict.get("answer", ""))
    return return_dict


def real_evaluate(solver):
    data_filename = "../datasets/drop_v0_dev.jsonl.gz"
    examples = load_drop(data_filename)[1:-1]  # first one and the last one is for few-shot examples
    random.seed(0)
    random.shuffle(examples)
    examples = examples[128:928]
    questions = [example['inputs'] for example in examples]
    answers = [example['targets'] for example in examples]

    print(f"problem length: {len(examples)}")
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
            extracted_answer = str(res.get("answer", "NO ANSWER IN DICTIONARY"))
            correct_answers = answers[q_idx]
            em_score, f1_score = drop_metric(extracted_answer, correct_answers)
        except Exception as e:
            info_list.append(f"Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
            acc_list.append(0)
            continue
        acc_list.append(f1_score)
        info_list.append(f"Sample {q_idx}:\n{task_queue[q_idx]}\nModel Output: {res}\nModel Answer: {extracted_answer}\nCorrect Answers: {correct_answers}\nF1 Score: {acc_list[-1]}\n")

    acc = sum(acc_list) / len(acc_list)
    interval = bootstrap_confidence_interval(acc_list)
    if acc > last_test_acc:
        open(f"result/drop_{round(acc, 4)}.txt", "w").writelines([interval] + info_list)
    return acc


class DROP_Task:
    def evaluate(self, solver):
        data_filename = "../datasets/drop_v0_dev.jsonl.gz"
        examples = load_drop(data_filename)[1:-1]  # first one and the last one is for few-shot examples
        random.seed(0)
        random.shuffle(examples)
        examples = examples[:128]
        import time
        # random.seed(time.time())
        # random.shuffle(examples)
        examples = examples[:20]
        questions = [example['inputs'] for example in examples]
        answers = [example['targets'] for example in examples]

        print(f"problem length: {len(examples)}")
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
                extracted_answer = str(res.get("answer", "NO ANSWER IN DICTIONARY"))
                correct_answers = answers[q_idx]
                em_score, f1_score = drop_metric(extracted_answer, correct_answers)
            except Exception as e:
                info_list.append(f"Valid Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
                acc_list.append(0)
                continue
            acc_list.append(f1_score)
            info_list.append(f"Valid Sample {q_idx}:\n{task_queue[q_idx]}\nModel Output: {res}\nModel Answer: {extracted_answer}\nCorrect Answers: {correct_answers}\nF1 Score: {acc_list[-1]}\n")

        valid_acc = sum(acc_list) / len(acc_list)
        print("Acc:", valid_acc)
        if valid_acc >= threshold:
            test_acc = real_evaluate(solver)
            feedback = f"Valid F1 Score: {valid_acc}\nTest F1 Score {test_acc}\n" + "Evaluation Info:\n" + "\n".join(info_list)
        else:
            test_acc = 0
            feedback = f"Valid F1 Score: {valid_acc}\nValid F1 Score less than {threshold}, no testing needed.\n" + "Evaluation Info:\n" + "\n".join(info_list)
        return feedback, test_acc


def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _answer_to_bags(
        answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
             (2 * precision * recall) / (precision + recall)
             if not (precision == 0.0 and recall == 0.0)
             else 0.0
         ) * 100
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_drop_metrics(
        predicted: Union[str, List[str], Tuple[str, ...]], gold: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return float(exact_match), float(f1)


def answer_json_to_strings(answer: Dict[str, Any]) -> Tuple[Tuple[str, ...], str]:
    """
    Takes an answer JSON blob from the DROP data release and converts it into strings used for
    evaluation.
    """
    if "number" in answer and answer["number"]:
        return tuple([str(answer["number"])]), "number"
    elif "spans" in answer and answer["spans"]:
        return tuple(answer["spans"]), "span" if len(answer["spans"]) == 1 else "spans"
    elif "date" in answer:
        return (
            tuple(
                [
                    "{0} {1} {2}".format(
                        answer["date"]["day"], answer["date"]["month"], answer["date"]["year"]
                    ).strip()
                ]
            ),
            "date",
        )
    else:
        raise ValueError(
            f"Answer type not found, should be one of number, spans or date at: {json.dumps(answer)}"
        )


def answer_json_to_string(answer_json):
    return json.dumps(answer_json_to_strings(answer_json))


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1


def drop_metric(sample: str, reference: list[str]) -> Tuple[float, float]:
    em_scores = []
    f1_scores = []
    for answer in reference:
        if answer.strip() != "":
            em, f1 = get_drop_metrics(sample, answer)
            em_scores.append(em)
            f1_scores.append(f1)
    return (max(em_scores), max(f1_scores))


def load_drop(file_path):
    with gzip.open(file_path, mode="rb") as f:
        test_samples = [json.loads(line) for line in f]
    prompt = """You will be asked to read a passage and answer a question.\n"""
    few_shot_prompt = """You will be asked to read a passage and answer a question.

# Examples:
Passage: As of the census of 2000, there were 952 people, 392 households, and 241 families residing in the village. The population density was 952.9 people per square mile (367.6/km²). There were 449 housing units at an average density of 449.4 per square mile (173.4/km²). The racial makeup of the village was 96.11% White (U.S. Census), 0.95% African American (U.S. Census) or Race (United States Census), 0.11% Native American (U.S. Census), 0.11% Asian (U.S. Census), 0.21% from Race (United States Census), and 2.52% from two or more races. 1.05% of the population were Hispanics in the United States or Latino (U.S. Census) of any race.\nQuestion: How many more people, in terms of percentage, were from two or more races compared to being solely Native American or solely Asian?\nAnswer: 2.3

# Your Task
---

"""
    examples = []
    for sample in test_samples:
        sample['inputs'] = few_shot_prompt + sample['context']
        sample['targets'] = sample["ref_text"].split("|")
        examples.append(sample)
    return examples


def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


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
    ci_lower_percent = ci_lower
    ci_upper_percent = ci_upper
    median_percent = median

    # Return the formatted string with confidence interval and median
    print(f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%")
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"
