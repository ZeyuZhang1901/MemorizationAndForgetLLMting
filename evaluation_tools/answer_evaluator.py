import pandas as pd
import openai
from tqdm import tqdm
from typing import List, Dict
import re

def evaluate_answers(api_key: str, model_name: str, answers: List[Dict]) -> pd.DataFrame:
    '''
    Evaluate the generated answers using a judge model.
    '''
    client = openai.OpenAI(api_key=api_key)
    results = []

    system_prompt = """You are an expert evaluator for question-answering systems. 
Your task is to evaluate the quality of generated answers compared to oracle answers.
You must output your evaluation in the exact format specified, with no additional text.
Your response must contain exactly one <score> tag and one <reason> tag."""

    for answer_dict in tqdm(answers, desc="Evaluating answers"):
        user_prompt = f"""Compare the following generated answer to the oracle answer and evaluate its quality:

Question: {answer_dict['question']}
Generated Answer: {answer_dict['model_answer']}
Oracle Answer: {answer_dict['oracle_answer']}

Rate the answer's quality on a scale of 0-100 based on:
- Correctness of information
- Completeness of response
- Relevance to the question

Provide your evaluation in exactly this format:
<score>
[single number between 0-100]
</score>

<reason>
[Your explanation without using words 'score' or 'reason']
</reason>"""

        # Make API call with system prompt
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3  # Lower temperature for more consistent formatting
        )

        evaluation = response.choices[0].message.content
        
        # Strict pattern matching
        score_pattern = r'<score>\s*(\d+(?:\.\d+)?)\s*</score>'
        reason_pattern = r'<reason>\s*(.*?)\s*</reason>'
        
        score_match = re.search(score_pattern, evaluation, re.DOTALL)
        reason_match = re.search(reason_pattern, evaluation, re.DOTALL)
        
        if not score_match or not reason_match:
            print(f"Invalid format detected in response: {evaluation}")
            print("Retrying evaluation...")
            # You could implement retry logic here
            continue
            
        try:
            score = float(score_match.group(1))
            if not (0 <= score <= 100):
                print(f"Score {score} out of range, retrying...")
                continue
        except ValueError:
            print(f"Invalid score format: {score_match.group(1)}, retrying...")
            continue
            
        reason = reason_match.group(1).strip()
        
        # Verify reason doesn't contain forbidden words
        if 'score' in reason.lower() or 'reason' in reason.lower():
            print("Found forbidden words in reason, retrying...")
            continue

        results.append({
            'Index': len(results),
            'Question': answer_dict['question'],
            'Generated Answer': answer_dict['model_answer'],
            'True Answer': answer_dict['oracle_answer'],
            'Score': score,
            'Reason': reason,
            'Article ID': answer_dict['article_id']
        })

    return pd.DataFrame(results)
