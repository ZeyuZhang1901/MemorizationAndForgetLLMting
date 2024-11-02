import pandas as pd
import openai
from tqdm import tqdm
import os
import json

filter_prompt = """You are given a question-answer pair from a news article. Your task is to evaluate the question based on the following criteria:
            
    1. **Specificity**: Does the question include specific details such as dates, names, events, or statistics?
    2. **Future Relevance**: Is the question framed in a way that it remains relevant and understandable in the future, even without the original article for context?
    3. **Fact-Based**: Is the question based on concrete facts rather than predictions or expectations?
    4. **Trivia Suitability**: Is the question suitable for a trivia game, meaning it should be clear, concise, and fact-based?
    
    Evaluate the question-answer pair and assign a score:
    - **Score: 1** if the question meets all the above criteria and is considered high-quality.
    - **Score: 0** if the question does not meet all the criteria and is considered low-quality.
    
    **Output Format**: Only output the score in the format “Score: X” where X is either 1 or 0. Do not include any additional text or explanation.
    
    Example of a high-quality question:
    - “As of October 2024, what significant adjustment is Starbucks making regarding its menu offerings?” (Score: 1)
    Example of a low-quality question:
    - “Which sectors are expected to be particularly affected by hurricanes in terms of job growth?” (Score: 0)
    
    Please evaluate the given question-answer pair and provide the score in the specified format. The question and answer are as follows:
    
    Question: {question_text}
    
    Answer: {answer_text}
    
    Please provide only your score and nothing else. If you provide any other text, you will break our downstream regex parser.
"""

def generate_questions(api_key, model_name, news_file, output_file):
    client = openai.OpenAI(api_key=api_key)
    
    news_df = pd.read_csv(news_file)
    qa_data = []

    for index, row in tqdm(news_df.iterrows(), total=len(news_df)):

        generate_prompt = f"""Suppose you are expert in journalist, Now given the following news and its information of topic, article, and date, you are asked to generate question-answer pairs. Specifically,
        
        The topic of the news is: {row['Topic']}
        
        The input article is: {row['Content']}
        
        The current date is: {row['Date']}
        
        You can assume the article was published around the current date, and that it's discussing current information unless otherwise specified.
        
        Please generate a question-answer pair from the article that includes specific details such as dates, names, events, or statistics. 
        
        The question should be framed in a way that it remains relevant and understandable in the future, even without the original article for context.
        
        Enough details should be included in the question so that the answer is objective and precise, especially those time-sensitive questions.
        
        For example, instead of asking "Who is the new chief global brand officer at Starbucks?", ask "Who took over as Starbucks global brand officer in October of 2024?"
        Similarly, instead of "Who is affected by the ongoing Boeing strike?", ask "In October 2024, how many workers were impacted by the Boeing strike?"
        
        Ensure that the question and answer pair could be used in a trivia game in 10 years and still make sense. 
        
        The answer should be concise and direct, better in one phrase or a short sentence.
        
        Now please generate 5 question-answer pairs from the article in the format of "Question1: question_text\n\nAnswer1: answer_text\n\nQuestion2: question_text\n\nAnswer2: answer_text\n\n...Question5: question_text\n\nAnswer5: answer_text"
        """

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": generate_prompt}]
        )

        qa_text = response.choices[0].message.content
        qa_pairs = []
        count = 0  # count the number of reasonable question-answer pairs for this news
        for i in range(1, 6):
            question_start = f"Question{i}:"
            answer_start = f"Answer{i}:"
            next_question_start = f"Question{i+1}:" if i < 5 else None

            question_index = qa_text.find(question_start)
            answer_index = qa_text.find(answer_start)
            next_question_index = qa_text.find(next_question_start) if next_question_start else len(qa_text)

            if question_index != -1 and answer_index != -1:
                question = qa_text[question_index + len(question_start):answer_index].strip()
                answer = qa_text[answer_index + len(answer_start):next_question_index].strip()
                # filter out the unreasonable question-answer pairs with the filter_prompt
                filter_response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": filter_prompt.format(question_text=question, answer_text=answer)}]
                )
                score = int(filter_response.choices[0].message.content.split(":")[1].strip())
                if score == 1:
                    qa_pairs.extend([question, answer])
                    count += 1
        print(f"Number of reasonable question-answer pairs: {count}")  # Add this line for debugging
        print(f"Success rate for news {index}: {count / 5}")

        for i in range(0, len(qa_pairs), 2):
            question = qa_pairs[i]
            answer = qa_pairs[i+1] if i+1 < len(qa_pairs) else "N/A"
            qa_data.append({'Index': index, 'Question': question, 'Answer': answer})

    qa_df = pd.DataFrame(qa_data)
    qa_df.to_csv(output_file, index=False)
    print(f"Questions and answers saved to {output_file}")

if __name__ == "__main__":
    with open('apikeys.json', 'r') as file:
        apikeys = json.load(file)
    api_key = apikeys["openai_api_key"]
    model_name = "gpt-4o"  # chatgpt-4o-mini
    news_file = "./data/news_articles/news_articles.csv"
    output_file = "./data/news_articles/evaluation_news_qa.csv"
    
    generate_questions(api_key, model_name, news_file, output_file)
