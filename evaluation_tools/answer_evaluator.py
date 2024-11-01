import pandas as pd
import openai
from tqdm import tqdm
import numpy as np

def evaluate_answers(api_key, model_name, news_file, qa_file, generated_answers, epoch):
    '''
    Evaluate the generated answers by the trained model.

    Args:
        api_key (str): The API key for OpenAI.
        model_name (str): The name of the model to evaluate.
        news_file (str): The path to the news file.
        qa_file (str): The path to the QA file.
        generated_answers (list): The generated answers by the model.
        epoch (int): The epoch number.

    Returns:
        dict: A dictionary containing the average score, median score, standard deviation, minimum score, maximum score, and detailed results.
    '''
    
    openai.api_key = api_key
    
    news_df = pd.read_csv(news_file)
    qa_df = pd.read_csv(qa_file)
    
    results = []

    for index, row in tqdm(qa_df.iterrows(), total=len(qa_df)):
        news_text = news_df.loc[row['Index'], 'Topic'] + "\n" + news_df.loc[row['Index'], 'Content']
        question = row['Question']
        true_answer = row['Answer']
        generated_answer = generated_answers[index]

        prompt = f"""Suppose you are expert in journalist, and you already know the news:

{news_text}

Then here are the generated answer by the trained model and the true answer (oracle):

Generated Answer: {generated_answer}

True Answer: {true_answer}

Please use your knowledge and the given news text to score the generated answer. Give a scalar reward out of 100, and provide the reason. Your response should be in this exact format:

'''Score: {{score}}

Reason: The reason of this score is because ...'''

Replace {{score}} with your numerical score, and complete the reason sentence."""

        response = openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        evaluation = response.choices[0].message.content
        # extract the score and reason from the evaluation
        score_start = evaluation.find('Score:') + len('Score:')
        reason_start = evaluation.find('Reason:')
        score = float(evaluation[score_start:reason_start].strip())
        reason = evaluation[reason_start + len('Reason:'):].strip()

        results.append({
            'Question': question,
            'Generated Answer': generated_answer,
            'True Answer': true_answer,
            'Score': score,
            'Reason': reason
        })

    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    average_score = results_df['Score'].mean()
    median_score = results_df['Score'].median()
    std_score = results_df['Score'].std()
    min_score = results_df['Score'].min()
    max_score = results_df['Score'].max()
    
    evaluation_results = {
        'average_score': average_score,
        'median_score': median_score,
        'std_score': std_score,
        'min_score': min_score,
        'max_score': max_score,
        'detailed_results': results
    }
    
    return evaluation_results
