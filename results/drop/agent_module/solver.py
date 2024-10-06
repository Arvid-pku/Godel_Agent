

def solver(agent_instance, task_input: str):
    messages = [
        {'role': 'system', 'content': 'You are a highly capable read comprehension expert.'},
        {'role': 'user', 'content': f'# Your Task:\n{task_input}'}
    ]
    response = agent_instance.action_call_json_format_llm(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=0.5,
        role='read comprehension expert',
        num_of_response=5,
        return_dict_keys=['reasoning', 'answer'],
        requirements='1. Provide explicit, step-by-step reasoning to reach the answer for clear understanding.\n2. Directly answer the question based on the passage and reasoning.\n3. The answer MUST be a concise string with no extra information.\n4. Ensure diverse, high-quality output for multiple attempts.'
    )
    # Applying consistency check and final output selection
    diverse_responses = response
    # Apply self-consistency: select most frequent or most detailed answer among diverse responses
    most_frequent_answer = max(set([resp['answer'] for resp in diverse_responses]), key=[resp['answer'] for resp in diverse_responses].count)
    final_response = next(resp for resp in diverse_responses if resp['answer'] == most_frequent_answer)
    final_response['answer'] = str(final_response.get('answer', ''))
    return final_response
