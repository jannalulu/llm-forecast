import json
import os
import requests
import re
from asknews_sdk import AskNewsSDK
import time
from asknews_sdk.errors import APIError
import sys
import logging
import datetime
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

log_filename = f"annabot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/annabot_newsonnet.log", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

args = sys.argv
if len(args) > 1 and args[1] == "dryrun":
    print("dry run, doing nothing")
    exit(0)

METACULUS_TOKEN = os.environ.get('METACULUS_TOKEN')
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ASKNEWS_CLIENT_ID = os.environ.get('ASKNEWS_CLIENT_ID')
ASKNEWS_SECRET = os.environ.get('ASKNEWS_SECRET')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api2"
TOURNAMENT_ID = 32506

# List questions and details

def setup_question_logger(question_id, log_type):
    """Set up a logger for a specific question and log type."""
    log_filename = f"logs/{question_id}_{log_type}.log"
    logger = logging.getLogger(f"{question_id}_{log_type}")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def log_question_reasoning(question_id, reasoning, question_title, model_name, run_number):
    """Log the reasoning for a specific question and run."""
    logger = setup_question_logger(question_id, "reasoning")
    logger.info(f"Question: {question_title}")
    logger.info(f"Reasoning for question {question_id}:\n{reasoning}")
    
    # JSON logging
    today = datetime.datetime.now().strftime('%Y%m%d')
    json_filename = f"logs/reasoning_{today}.json"
    
    question_data = {
        "question_id": question_id,
        "question_title": question_title,
        f"{model_name}_reasoning{run_number}": reasoning
    }
    
    try:
        # Read existing data if file exists
        if os.path.exists(json_filename):
            with open(json_filename, 'r', encoding='utf-8') as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = []
        
        # Update existing entry or add new one
        existing_entry = next((item for item in existing_data if item["question_id"] == question_id), None)
        if existing_entry:
            existing_entry.update(question_data)
        else:
            existing_data.append(question_data)
        
        # Write updated data
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"Error writing to {json_filename}: {str(e)}")

def log_question_news(question_id, news, question_title):
    """Log the news articles for a specific question."""
    # Standard logging
    logger = setup_question_logger(question_id, "news")
    logger.info(f"Question: {question_title}")
    logger.info(f"News articles for question {question_id}:\n{news}")
    
    # JSON logging
    today = datetime.datetime.now().strftime('%Y%m%d')
    json_filename = f"logs/news_{today}.json"
    
    news_data = {
        "question_id": question_id,
        "question_title": question_title,
        "news": news
    }
    
    try:
        # Read existing data if file exists
        if os.path.exists(json_filename):
            with open(json_filename, 'r', encoding='utf-8') as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = []
        
        # Update existing entry or add new one
        existing_entry = next((item for item in existing_data if item["question_id"] == question_id), None)
        if existing_entry:
            existing_entry.update(news_data)
        else:
            existing_data.append(news_data)
        
        # Write updated data
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"Error writing to {json_filename}: {str(e)}")

def list_questions(tournament_id=TOURNAMENT_ID, offset=0, count=None):
    """
    List open questions from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "has_group": "false",
        "order_by": "-activity",
        "forecast_type": "binary",
        "project": tournament_id,
        "status": "open",
        "type": "forecast",
        "include_description": "true",
    }
    if count is not None:
        url_qparams["limit"] = count
    url = f"{API_BASE_URL}/questions/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)
    response.raise_for_status()
    data = json.loads(response.content)
    return data

def asknews_api_call_with_retry(func, *args, **kwargs):
    max_retries = 5
    base_delay = 1
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            if e.error_code == 500000:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logging.warning(f"AskNews API Internal Server Error. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error("AskNews API Internal Server Error persisted after max retries.")
                    raise
            else:
                raise

def get_formatted_asknews_context(query):
    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID,
        client_secret=ASKNEWS_SECRET,
        scopes=["news"]
    )

    try:
        # get the latest news related to the query (within the past 48 hours)
        hot_response = asknews_api_call_with_retry(
            ask.news.search_news,
            query=query,  # your natural language query
            n_articles=5,  # control the number of articles to include in the context
            return_type="both",
            strategy="latest news"  # enforces looking at the latest news only
        )

        # get context from the "historical" database that contains a news archive going back to 2023
        historical_response = asknews_api_call_with_retry(
            ask.news.search_news,
            query=query,
            n_articles=25,
            return_type="both",
            strategy="news knowledge"  # looks for relevant news within the past 60 days
        )

        formatted_articles = format_asknews_context(
            hot_response.as_dicts, historical_response.as_dicts)
    except APIError as e:
        logging.error(f"AskNews API error: {e}")
        formatted_articles = "Error fetching news articles. Please try again later."

    return formatted_articles

def format_asknews_context(hot_articles, historical_articles):
  formatted_articles = "Here are the relevant news articles:\n\n"

  if hot_articles:
    hot_articles = [article.__dict__ for article in hot_articles]
    hot_articles = sorted(
        hot_articles, key=lambda x: x['pub_date'], reverse=True)

    for article in hot_articles:
        pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
        formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

  if historical_articles:
    historical_articles = [article.__dict__ for article in historical_articles]
    historical_articles = sorted(
        historical_articles, key=lambda x: x['pub_date'], reverse=True)

    for article in historical_articles:
        pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
        formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

  if not hot_articles and not historical_articles:
    formatted_articles += "No articles were found.\n\n"
    return formatted_articles

  formatted_articles += f"*Generated by AI at [AskNews](https://asknews.app), check out the [API](https://docs.asknews.app) for more information*."
  logging.info(f"News articles:\n{formatted_articles}")
  return formatted_articles

# Prompt
PROMPT_TEMPLATE = """
You are a superforecaster who has a strong track record of accurate forecasting. You evaluate past data and trends carefully for potential clues to future events, while recognising that the past is an imperfect guide to the future so you will need to put probabilities on possible future outcomes (ranging from 0 to 100%). Your specific goal is to maximize the accuracy of these probability judgments by minimising the Brier scores that your probability judgments receive once future outcomes are known.
Brier scores have two key components:
1. calibration (across all questions you answer, the probability estimates you assign to possible future outcomes should correspond as closely as possible to the objective frequency with which outcomes occur).
2. resolution (across all questions, aim to assign higher probabilities to events that occur than to events that do not occur).

The question that you are forecasting as well as some background information and resolution criteria are below. 

Your question is:
{title}

The Resolution Criteria for the question is:
{resolution_criteria}

You found the following news articles related to the question:
{formatted_articles}

background:
{background}

fine print:
{fine_print}

Today is {today}.

Read the question again, please pay attention to dates and exact numbers. Work through each step before making your prediction. Double-check whether your prediction makes sense before stating ZZ.ZZ% is the most likely.
Carefully outline your reasons for each forecast: list the strongest evidence and arguments for making lower or higher estimates and explain how you balance the evidence to make your own forecast. You begin this analytic process by looking for reference or comparison classes of similar events and grounding your initial estimates in base rates of occurrence (how often do events of this sort occur in situations that look like the present one?). You then adjust that initial estimate in response to the latest news and distinctive features of the present situation, recognising the need for flexible adjustments but also the risks of over-adjusting and excessive volatility. Superforecasting requires weighing the risks of opposing errors: e.g., of failing to learn from useful historical patterns vs. over-relying on misleading patterns. In this process of error balancing, you draw on the 10 commandments of superforecasting (Tetlock & Gardner, 2015) as well as on other peer-reviewed research on superforecasting.
1. Triage and reference relevant predictions from humans if they exist, such as FiveThirtyEight, Polymarket, and Metaculus.
2. Break seemingly intractable problems into tractable sub-problems.
3. Strike the right balance between inside and outside views.
4. Strike the right balance between under- and overreacting to evidence.
5. Look for the clashing causal forces at work in each problem.
6. Extrapolate current the trends linearly.
7. Strive to distinguish as many degrees of doubt as the problem permits but no more.
8. Strike the right balance between under- and overconfidence, between prudence and decisiveness.
9. Look for the errors behind your mistakes but beware of rearview-mirror hindsight biases.

Once you have written your reasons, ensure that they directly inform your forecast; please make sure that you're answering the {title}. Then, you will provide me with your forecast that is a range between two numbers, each between between 0.10 and 99.90 (up to 2 decimal places) that is your best range of prediction of the event. 
Output your prediction as “My Prediction: Between XX.XX% and YY.YY%, but ZZ.ZZ% being the most likely. Probability: ZZ.ZZ%." Please not add anything after. 

"""

#GPT-4 predictions
def get_summary_from_gpt(all_runs_text):
    url = "https://www.metaculus.com/proxy/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Token {METACULUS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": f"Please provide a concise summary of these forecasting runs, focusing on the key points of reasoning and how they led to the probabilities. You must include the probabilities from each run. Here are the runs:\n\n{all_runs_text}"
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    except requests.RequestException as e:
        print(f"Error getting summary: {e}")
        return None

def get_gpt_prediction(question_details, formatted_articles):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # client = OpenAI(api_key=OPENAI_API_KEY)

    prompt_input = {
        "title": question_details["question"]["title"],
        "background": question_details["question"]["description"],
        "resolution_criteria": question_details["question"].get("resolution_criteria", ""),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today
    }

    url = "https://www.metaculus.com/proxy/openai/v1/chat/completions/"
    
    headers = {
        "Authorization": f"Token {METACULUS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(**prompt_input)
            }
        ]
    }
    
    max_retries = 10
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            gpt_text = response_data['choices'][0]['message']['content']
            return gpt_text
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"GPT API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}")
                time.sleep(delay)
            else:
                logging.error(f"GPT API error persisted after {max_retries} retries: {e}")
                return None

def get_claude_prediction(question_details, formatted_articles):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt_input = {
        "title": question_details["question"]["title"],
        "background": question_details["question"]["description"],
        "resolution_criteria": question_details["question"].get("resolution_criteria", ""),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today
    }

    url = "https://www.metaculus.com/proxy/anthropic/v1/messages/"

    headers = {
        "Authorization": f"Token {METACULUS_TOKEN}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }

    data = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(**prompt_input)
            }
        ]
    }
    
    max_retries = 10
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            response_data = response.json()
            claude_text = response_data['content'][0]['text']
            return claude_text
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"Claude API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}")
                time.sleep(delay)
            else:
                logging.error(f"Claude API error persisted after {max_retries} retries: {e}")
                return None

# Find all numbers followed by a '%'
def find_number_before_percent(s):
    matches = re.findall(r'(\d+(?:\.\d{1,2})?)%', s)
    if matches:
        return float(matches[-1])
    else:
        return None
    
def extract_probability(ai_text):
    probability_match = find_number_before_percent(ai_text)

    # Extract the number if a match is found
    probability = None
    if probability_match:
        probability = float(probability_match) # int(match.group(1))
        logging.info(f"The extracted probability is: {probability}%")
        probability = min(max(probability, 1), 99) # To prevent extreme forecasts
        return probability
    else:
        print("Unable to extract probability.")
        return None

#GPT prediction and submitting a forecast

questions = list_questions() #find open questions

open_questions_ids = []

for question in questions["results"]:
    if question["status"] == "open":
        print(f"ID: {question['id']}\nQ: {question['title']}\nCloses: {question['scheduled_close_time']}")
        open_questions_ids.append(question["id"])

def post_question_comment(question_id, comment_text):
    """
    Post a comment on the question page as the bot user.
    """
    response = requests.post(
        f"{API_BASE_URL}/comments/",
        json={
            "comment_text": comment_text,
            "submit_type": "N",
            "include_latest_prediction": True,
            "question": question_id,
        },
        **AUTH_HEADERS,
    )
    response.raise_for_status()

def post_question_prediction(question_id, prediction_percentage):
    """
    Post a prediction value (between 1 and 100) on the question.
    """
    url = f"{API_BASE_URL}/questions/{question_id}/predict/"
    response = requests.post(
        url,
        json={"prediction": float(prediction_percentage) / 100},
        **AUTH_HEADERS,
    )
    response.raise_for_status()


def log_predictions_json(question_id, question_title, gpt_results, claude_results, gpt_texts, claude_texts, average_probability):
    """Log predictions and reasoning to a JSON file."""
    json_filename = "logs/reasoning_{today}.json"
    
    prediction_data = {
        "question_id": question_id,
        "question_title": question_title,
        "timestamp": datetime.datetime.now().isoformat(),
        "runs": [],
        "average_probability": average_probability
    }
    
    for i in range(len(gpt_results)):
        prediction_data["runs"].append({
            "run_number": i + 1,
            "gpt_prediction": gpt_results[i],
            "gpt_reasoning": gpt_texts[i],
            "claude_prediction": claude_results[i],
            "claude_reasoning": claude_texts[i]
        })
    
    try:
        if os.path.exists(json_filename):
            with open(json_filename, 'r', encoding='utf-8') as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = []
            
        # Update existing entry or add new one
        existing_entry = next((item for item in existing_data if item["question_id"] == question_id), None)
        if existing_entry:
            existing_entry.update(prediction_data)
        else:
            existing_data.append(prediction_data)
            
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=2)
            
        logging.info(f"Successfully logged predictions for question {question_id} to {json_filename}")
    except Exception as e:
        logging.error(f"Error writing to {json_filename}: {str(e)}")

def get_question_details(question_id):
    """
    Get all details about a specific question.
    """
    url = f"{API_BASE_URL}/questions/{question_id}/"
    response = requests.get(
        url,
        **AUTH_HEADERS,
    )
    response.raise_for_status()
    return json.loads(response.content)

SUBMIT_PREDICTION = True

#GPT prediction and submitting a forecast

questions = list_questions() #find open questions

open_questions_ids = []
for question in questions["results"]:
    if question["status"] == "open":
        print(f"ID: {question['id']}\nQ: {question['title']}\nCloses: {question['scheduled_close_time']}")
        open_questions_ids.append(question["id"])

for question_id in open_questions_ids:
    print(f"Question id: {question_id}\n\n")
    question_details = get_question_details(question_id)
    print("Question details:\n\n", question_details)

    formatted_articles = get_formatted_asknews_context(question_details["title"])
    log_question_news(question_id, formatted_articles, question_details["question"]["title"])
    gpt_probabilities = []
    claude_probabilities = []
    gpt_texts = []
    claude_texts = []
    
    for run in range(5):
        print(f"Run {run} for question {question_id}")
        
        gpt_result = get_gpt_prediction(question_details, formatted_articles)
        claude_result = get_claude_prediction(question_details, formatted_articles)
        
        if gpt_result:
            gpt_texts.append(gpt_result)
        if claude_result:
            claude_texts.append(claude_result)
            
        log_question_reasoning(question_id, gpt_result, question_details["question"]["title"], "gpt", run)
        log_question_reasoning(question_id, claude_result, question_details["question"]["title"], "claude", run)

        gpt_probability = extract_probability(gpt_result)
        claude_probability = extract_probability(claude_result)
        
        if gpt_probability is not None:
            gpt_probabilities.append(gpt_probability)
        if claude_probability is not None:
            claude_probabilities.append(claude_probability)

    if gpt_probabilities and claude_probabilities:
        gpt_avg = sum(gpt_probabilities) / len(gpt_probabilities)
        claude_avg = sum(claude_probabilities) / len(claude_probabilities)
        average_probability = (gpt_avg + claude_avg) / 2
        logging.info(f"GPT probabilities: {gpt_probabilities}")
        logging.info(f"GPT average: {gpt_avg}%")
        logging.info(f"Claude probabilities: {claude_probabilities}")
        logging.info(f"Claude average: {claude_avg}%")
        logging.info(f"Overall average: {average_probability}%")

        # Get summary of all runs
        summary_prompt = f"Analyze and summarize these forecasting runs:\n\nFirst group of runs:\n{gpt_result}\n\nSecond group of runs:\n{claude_result}\n\nProbabilities from all runs: {gpt_probabilities + claude_probabilities}. Write out the probabilities of all the runs in one sentence. Keep your summary of the key points that each forecast brought up in under 150 words."
        
        summary_data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": summary_prompt}]
        }
        
        url = "https://www.metaculus.com/proxy/openai/v1/chat/completions/"
        headers = {
            "Authorization": f"Token {METACULUS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, headers=headers, json=summary_data)
            response.raise_for_status()
            summary = response.json()['choices'][0]['message']['content']

            summary_data = {
                "question_id": question_id,
                "question_title": question_details["question"]["title"],
                "summary": summary
            }
            
            today = datetime.datetime.now().strftime('%Y%m%d')
            json_filename = f"logs/reasoning_{today}.json"
            
            try:
                if os.path.exists(json_filename):
                    with open(json_filename, 'r', encoding='utf-8') as json_file:
                        existing_data = json.load(json_file)
                else:
                    existing_data = []
                
                existing_entry = next((item for item in existing_data if item["question_id"] == question_id), None)
                if existing_entry:
                    existing_entry["summary"] = summary
                else:
                    existing_data.append({
                        "question_id": question_id,
                        "question_title": question_details["question"]["title"],
                        "summary": summary
                    })
                
                with open(json_filename, 'w', encoding='utf-8') as json_file:
                    json.dump(existing_data, json_file, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                logging.error(f"Error writing summary to {json_filename}: {str(e)}")
            
            if SUBMIT_PREDICTION:
                post_question_prediction(question_id, average_probability)
                comment = f"Summary of 5 runs:\n\n{summary}\n\nFinal prediction: {average_probability:.2f}%"
                print(f"Posting comment: {comment}\n\n")
                post_question_comment(question_id, comment)

        except Exception as e:
            logging.error(f"Error getting summary: {e}")
    else:
        print("Unable to extract probability.")