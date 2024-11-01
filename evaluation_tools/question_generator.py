import pandas as pd
import openai
from tqdm import tqdm
import os
import json
def generate_questions(api_key, model_name, news_file, output_file):
    client = openai.OpenAI(api_key=api_key)
    
    news_df = pd.read_csv(news_file)
    qa_data = []

    for index, row in tqdm(news_df.iterrows(), total=len(news_df)):
        prompt = f"""Suppose you are expert in journalist, Now given the following news and its information in the format of "topic", "content", and "keywords", please generate 5 questions with the oracle answers of each of them. Answer in the following format: Question1: question_text\n\nAnswer1: answer_text\n\nQuestion2: question_text\n\nAnswer2: answer_text\n\n...Question5: question_text\n\nAnswer5: answer_text

Topic: {row['Topic']}

Content: {row['Content']}

Keywords: {row['Key_words']}

Make sure the questions are related to the news and the answers are correct. There should be 5 questions and 5 answers separated by '\n\n' so that we can easily split them."""

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        qa_text = response.choices[0].message.content
        qa_pairs = []
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
                qa_pairs.extend([question, answer])
        print(f"Number of qa_pairs: {len(qa_pairs)}")  # Add this line for debugging
        print(qa_pairs)

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
    model_name = "gpt-4-turbo"  # chatgpt-4o-mini
    news_file = "./data/news_articles/news_articles.csv"
    output_file = "./data/news_articles/evaluation_news_qa.csv"
    
    generate_questions(api_key, model_name, news_file, output_file)
