import pandas as pd
import openai
from tqdm import tqdm
import json
import os

def generate_dpo_data(api_key, model_name, news_file, output_file, num_questions=20):
    client = openai.OpenAI(api_key=api_key)
    news_df = pd.read_csv(news_file)
    
    # Load existing data if file exists
    existing_data = []
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        existing_data = existing_df.to_dict('records')
        print(f"Loaded {len(existing_data)} existing examples")
    
    new_data = []

    filter_prompt = """You are given a question from a news article. Your task is to evaluate the question based on the following criteria:
            
    1. **Specificity**: Does the question include specific details such as dates, names, events, or statistics?
    2. **Future Relevance**: Is the question framed in a way that it remains relevant and understandable in the future, even without the original article for context?
    3. **Fact-Based**: Is the question based on concrete facts rather than predictions or expectations?
    4. **Trivia Suitability**: Is the question suitable for a trivia game, meaning it should be clear, concise, and fact-based?
    
    Evaluate the question and assign a score:
    - **Score: 1** if the question meets all the above criteria and is considered high-quality.
    - **Score: 0** if the question does not meet all the criteria and is considered low-quality.
    
    Example of a high-quality question:
    - “As of October 2024, what significant adjustment is Starbucks making regarding its menu offerings?” (Score: 1)
    Example of a low-quality question:
    - “Which sectors are expected to be particularly affected by hurricanes in terms of job growth?” (Score: 0)
    
    Please evaluate the given question and provide the score in the specified format. The question is in the following format:
    
    Question: question_text
    
    **Output Format**: Only output the score in the format "Score: X" where X is either 1 or 0. Do not include any additional text or explanation."""

    system_prompt = """You are an expert journalist and educator specialized in creating high-quality question-answer pairs for AI model training. Your expertise lies in:
    1. Crafting specific, fact-based questions that remain relevant over time
    2. Writing clear, verifiable correct answers
    3. Creating plausible but factually incorrect alternative answers
    
    For each question you generate, you must follow these strict requirements:

    QUESTION REQUIREMENTS:
    - Include specific details (dates, names, events, statistics)
    - Add temporal context for time-sensitive information
    - Ensure future relevance without requiring the original article
    - Make questions suitable for trivia games
    - Focus on objective, verifiable facts
    
    CORRECT ANSWER REQUIREMENTS:
    - Keep answers concise (one short sentence)
    - Include precise, verifiable details
    - Avoid opinions, predictions, or subjective statements
    - Maintain factual accuracy from the source
    
    WRONG ANSWER REQUIREMENTS:
    - Create plausible but factually incorrect answers
    - Maintain the same topic and context as the correct answer
    - Ensure the error is objectively verifiable
    - Match the style and length of the correct answer
    - Make errors specific (e.g., wrong numbers, dates, names) rather than vague

    Your output must strictly follow this format for each set:
    SET[number]
    Question: [question_text]
    Correct: [correct_answer_text]
    Wrong: [wrong_answer_text]"""

    user_prompt = """Generate {num_questions} high-quality question-answer sets based on this news article:

    Topic: {topic}
    Date: {date}
    Content: {content}

    Requirements:
    1. Generate exactly {num_questions} different questions
    2. Cover different aspects of the article
    3. Follow the exact format specified
    4. Ensure each wrong answer contains a specific, verifiable error

    Example of good format:
    SET1
    Question: Who took over as Starbucks global brand officer in October of 2024?
    Correct: Brady Brewer.
    Wrong: Michael Thompson.

    SET2
    Question: In October 2024, how many workers were impacted by the Boeing strike?
    Correct: 8,000 Boeing workers were affected.
    Wrong: 12,500 Boeing workers were affected.

    Please generate your {num_questions} sets now, maintaining consistent formatting and separating each set with a blank line."""

    for index, row in tqdm(news_df.iterrows(), total=len(news_df)):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(
                topic=row['Topic'],
                content=row['Content'],
                date=row['Date'],
                num_questions=num_questions
            )}
        ]

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )

            # Parse the response
            qa_text = response.choices[0].message.content
            qa_sets = qa_text.strip().split('\n\n')

            count = 0  # Count the number of high-quality questions
            for qa_set in qa_sets:
                if not qa_set.startswith('SET'):
                    continue
                    
                lines = [line.strip() for line in qa_set.split('\n') if line.strip()]
                if len(lines) >= 4:
                    try:
                        question = lines[1].replace('Question:', '').strip()
                        correct = lines[2].replace('Correct:', '').strip()
                        wrong = lines[3].replace('Wrong:', '').strip()
                        
                        if all([question, correct, wrong]):
                            filter_messages = [
                                {"role": "system", "content": filter_prompt},
                                {"role": "user", "content": f"Question: {question}"}
                            ]
                            
                            filter_response = client.chat.completions.create(
                                model=model_name,
                                messages=filter_messages
                            )
                            
                            score = filter_response.choices[0].message.content.strip()
                            if score == "Score: 1":
                                count += 1
                                new_entry = {
                                    'Input_Text': question,
                                    'Accepted_Completion': correct,
                                    'Rejected_Completion': wrong
                                }
                                # Check if this exact question doesn't exist in both existing and new data
                                if not any(entry['Input_Text'] == question for entry in existing_data + new_data):
                                    new_data.append(new_entry)
                    except IndexError:
                        print(f"Parsing error in set: {qa_set}")
                        continue

            print(f"Generated {count} high-quality questions from article {index + 1}")
            
        except Exception as e:
            print(f"Error processing article {index + 1}: {str(e)}")
            continue

    # Combine existing and new data
    all_data = existing_data + new_data
    
    # Save all data to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    
    print(f"\nGeneration Statistics:")
    print(f"Previously existing examples: {len(existing_data)}")
    print(f"Newly generated examples: {len(new_data)}")
    print(f"Total examples: {len(all_data)}")
    print(f"DPO training data saved to {output_file}")

if __name__ == "__main__":
    # Load API key
    with open('apikeys.json', 'r') as file:
        apikeys = json.load(file)
    api_key = apikeys["openai_api_key"]
    
    # Configuration
    model_name = "gpt-4o"
    news_file = "./data/news_articles/news_articles.csv"
    output_file = "./data/news_articles/news_articles_dpo.csv"
    
    generate_dpo_data(api_key, model_name, news_file, output_file, num_questions=20)