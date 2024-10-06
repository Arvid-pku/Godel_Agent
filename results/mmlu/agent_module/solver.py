

def solver(agent, task: str):
    # Few-Shot Learning: Providing extended examples to guide the LLM
    few_shot_examples = [
        {'role':'user', 'content':'Question: In the movie Austin Powers: The Spy Who Shagged Me what is the name of Dr. Evil\'s diminutive clone?\nChoices:\n(A) Little Buddy\n(B) Mini-Me\n(C) Small Fry\n(D) Dr Evil Jr'},
        {'role':'assistant', 'content':'In the movie Austin Powers: The Spy Who Shagged Me, Dr. Evil\'s diminutive clone is famously named Mini-Me.\nAnswer: B'},
        {'role':'user', 'content':'Question: What type of covalent bonds link the amino acids in a protein?\nChoices:\n(A) Peptide bonds\n(B) Hydrogen bonds\n(C) Ionic bonds\n(D) Glycosidic bonds'},
        {'role':'assistant', 'content':'Amino acids in a protein are linked by peptide bonds. Peptide bonds are formed through a dehydration synthesis reaction between the amino group of one amino acid and the carboxyl group of another amino acid.\nAnswer: A'},
        {'role':'user', 'content':'Question: Which writer was concerned with the reaction of workers to key characteristics of bureaucracies? Choices: (A) Merton (B) Weber (C) Gouldner (D) Mayo'},
        {'role':'assistant','content':'Max Weber was the writer who was concerned with the reaction of workers to key characteristics of bureaucracies.\nAnswer: B'},
        {'role':'user','content':'Sulfurous acid is a weak acid, while sulfuric acid is a much stronger acid because Choices:(A) the sulfur in sulfuric acid is more electronegative than the sulfur in sulfurous acid (B) sulfuric acid has more oxygen atoms in its formula (C) the O\u2013H bonds in sulfuric acid are much weaker than those in sulfurous acid (D) sulfurous acid has its hydrogen atoms bound directly to the sulfur atom'},
        {'role':'assistant','content':'Sulfuric acid is a stronger acid than sulfurous acid because it has more oxygen atoms in its formula, which leads to greater acidity.\nAnswer: B'},
        # Adding more diverse examples to our few-shot learning
        {'role':'user', 'content':'Question: Who developed the theory of relativity?\nChoices: (A) Isaac Newton (B) Albert Einstein (C) Galileo Galilei (D) Nikola Tesla'},
        {'role':'assistant', 'content':'Albert Einstein developed the theory of relativity.\nAnswer: B'},
        {'role':'user', 'content':'Question: What is the capital city of Japan?\nChoices: (A) Beijing (B) Seoul (C) Tokyo (D) Bangkok'},
        {'role':'assistant', 'content':'The capital city of Japan is Tokyo.\nAnswer: C'},
        {'role':'user', 'content':'Question: Lorem Ipsum?\nChoices: (A) Lorem\n(B) Ipsum\n(C) Dolor\n(D) Sit Amet'},
        {'role':'assistant', 'content':'Answer: A'}
    ]
    
    # Integrate the few-shot examples into the conversation
    messages = few_shot_examples + [{'role': 'user', 'content': f'# Your Task:\n{task}'}]
    
    # Using self-consistency by generating multiple responses
    response = agent.action_call_json_format_llm(
        model='gpt-3.5-turbo', 
        messages=messages, 
        temperature=0.8, 
        num_of_response=5,
        role='knowledge and reasoning expert', 
        return_dict_keys=['reasoning', 'answer'], 
        requirements=(
            '1. Please explain step by step.\n'
            '2. The answer MUST be either A or B or C or D.\n'
        ).strip(), 
    )
    
    # Select the most consistent response
    answer_frequency = {}
    for resp in response:
        answer = resp.get('answer', '')
        if answer in ['A', 'B', 'C', 'D']:
            if answer in answer_frequency:
                answer_frequency[answer] += 1
            else:
                answer_frequency[answer] = 1
    
    most_consistent_answer = max(answer_frequency, key=answer_frequency.get)
    consistent_response = next(resp for resp in response if resp.get('answer') == most_consistent_answer)
    consistent_response['answer'] = most_consistent_answer
    
    return consistent_response