

def solver(agent, task: str):
    from typing import List, Dict
    import json
    messages = [{"role": "user", "content": f"# Your Task:\n{task}"}]
    response = agent.action_call_json_format_llm(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,
        num_of_response=20,
        role="math expert",
        return_dict_keys=["reasoning", "answer"],
        requirements=(
            "1. Please explain step by step.\n"
            "2. The answer SHOULD be an integer.\n"
        ).strip(),
    )
    
    def parse_answers(response):
        answers = []
        for r in response:
            try:
                answer = str(int(r['answer']))  # Ensure the answer is an integer in string format
                answers.append(answer)
            except ValueError:
                continue
        return answers

    def find_most_consistent_answer(answers: List[str]) -> str:
        answer_count: Dict[str, int] = {}
        for ans in answers:
            if ans in answer_count:
                answer_count[ans] += 1
            else:
                answer_count[ans] = 1
        most_consistent_answer = max(answer_count, key=answer_count.get)
        return most_consistent_answer

    parsed_answers = parse_answers(response)
    most_consistent_answer = find_most_consistent_answer(parsed_answers)

    return_dict = response[0]
    return_dict["answer"] = most_consistent_answer
    return return_dict