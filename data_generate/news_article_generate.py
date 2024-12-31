import requests
from newspaper import Article, ArticleException
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import csv
import os
import random
import time
from requests.exceptions import RequestException
import json

# # Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('vader_lexicon')
# nltk.download('stopwords')

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
]

def fetch_news_articles(api_key, num_articles=100):
    base_url = "https://newsapi.org/v2/top-headlines"
    categories = ["business", "entertainment", "general", "health", "science", "sports", "technology"]
    articles = []

    for category in categories:
        params = {
            "apiKey": api_key,
            "category": category,
            "language": "en",
            "pageSize": 10,  # Maximum allowed by the free tier
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        articles.extend(data.get("articles", []))
        
        if len(articles) >= num_articles:
            break

    return articles[:num_articles]

def process_article(article_data, max_retries=3):
    url = article_data.get("url")
    if not url:
        return None

    for attempt in range(max_retries):
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            
            # Get the publish date
            publish_date = article.publish_date.strftime('%Y-%m-%d') if article.publish_date else 'Unknown'

            # Check article length (minimum 500 words)
            word_count = len(article.text.split())
            if word_count < 200:
                return None

            # Check if the article has a summary and keywords
            if not article.summary or not article.keywords:
                return None

            # Perform sentiment analysis
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(article.summary)

            # Determine the main topic
            topic = determine_topic(article.text)

            return {
                "title": article.title,
                "text": article.text,
                "summary": article.summary,
                "keywords": article.keywords,
                "sentiment": sentiment,
                "url": url,
                "topic": topic,
                "word_count": word_count,
                "date": publish_date
            }
        except ArticleException as e:
            print(f"ArticleException processing {url}: {e}")
            if "403" in str(e) or "Forbidden" in str(e):
                # If we get a 403 error, try changing the user agent
                article.config.browser_user_agent = random.choice(user_agents)
            time.sleep(2 ** attempt)  # Exponential backoff
        except RequestException as e:
            print(f"RequestException processing {url}: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            print(f"Error processing article {url}: {e}")
            return None

    print(f"Failed to process article after {max_retries} attempts: {url}")
    return None

def determine_topic(text):
    # Tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Get word frequencies
    word_freq = Counter(filtered_words)
    
    # Extract top words
    top_words = [word for word, _ in word_freq.most_common(10)]
    
    # Generate a summary using the top words
    summary = ' '.join(top_words)
    
    # Truncate to 20 words if necessary
    words = summary.split()
    if len(words) > 20:
        summary = ' '.join(words[:20])
    
    return summary

def is_interesting_and_diverse(article, existing_topics):
    # Define criteria for interesting and diverse articles
    is_new_topic = article["topic"] not in existing_topics
    has_keywords = len(article["keywords"]) >= 3
    is_informative = len(article["summary"].split()) >= 25
    has_sentiment = abs(article["sentiment"]["compound"]) > 0.1

    return is_new_topic and has_keywords and is_informative and has_sentiment

def main(api_key, num_articles=500):
    raw_articles = fetch_news_articles(api_key, num_articles)
    processed_articles = []
    existing_topics = set()

    for article_data in raw_articles:
        processed = process_article(article_data)
        if processed and is_interesting_and_diverse(processed, existing_topics):
            processed_articles.append(processed)
            existing_topics.add(processed["topic"])

        if len(processed_articles) >= num_articles:
            break

    print(f"Collected {len(processed_articles)} diverse and informative articles")
    return processed_articles

def save_to_csv(articles, folder_path, filename = "news_articles.csv"):
    print(f"Saving {len(articles)} articles to {folder_path}")
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, filename), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Index", "Topic", "Content", "Word_count", "Key_words", "Date"])

        writer.writeheader()
        for index, article in enumerate(articles, start=1):
            writer.writerow({
                'Index': index,
                'Topic': article['topic'],
                'Content': article['text'],
                'Word_count': article['word_count'],
                'Key_words': ', '.join(article['keywords']),
                'Date': article['date']
            })

    print(f"Articles saved to {filename}")

if __name__ == "__main__":
    with open('apikeys.json', 'r') as file:
        apikeys = json.load(file)
    API_KEY = apikeys["news_api_key"]
    articles = main(API_KEY)
    save_to_csv(articles, "./data/news_articles")
