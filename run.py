# -*- coding: utf-8 -*-

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

args = sys.argv
if len(args) > 1 and args[1] == "dryrun":
    print("dry run, doing nothing")
    exit(0)

## AI Forecasting AnnaBot

def read_secrets(path):
    secrets = {}
    with open(path) as f:
        for l in f:
            kv = l.strip().split('=', 1)
            k = kv[0]
            v = kv[1]
            secrets[k] = v
    return secrets

secrets = read_secrets("secrets.txt")

METACULUS_TOKEN = secrets['METACULUS_TOKEN']
# OPENAI_API_KEY = secrets['OPENAI_API_KEY']
ASKNEWS_CLIENT_ID = secrets['ASKNEWS_CLIENT_ID']
ASKNEWS_SECRET = secrets['ASKNEWS_SECRET']
# ANTHROPIC_API_KEY = secrets['ANTHROPIC_API_KEY']

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
1. Triage and reference relevant predictions from humans if they exist, such as FiveThirtyEight, Polymarket, and Metaculus.
2. Break seemingly intractable problems into tractable sub-problems.
3. Strike the right balance between inside and outside views.
4. Strike the right balance between under- and overreacting to evidence.
5. Look for the clashing causal forces at work in each problem.
6. Extrapolate current the trends linearly.
7. Strive to distinguish as many degrees of doubt as the problem permits but no more.
8. Strike the right balance between under- and overconfidence, between prudence and decisiveness.
9. Look for the errors behind your mistakes but beware of rearview-mirror hindsight biases.

Carefully outline your reasons for each forecast: list the strongest evidence and arguments for making lower or higher estimates and explain how you balance the evidence to make your own forecast. You begin this analytic process by looking for reference or comparison classes of similar events and grounding your initial estimates in base rates of occurrence (how often do events of this sort occur in situations that look like the present one?). You then adjust that initial estimate in response to the latest news and distinctive features of the present situation, recognising the need for flexible adjustments but also the risks of over-adjusting and excessive volatility. Superforecasting requires weighing the risks of opposing errors: e.g., of failing to learn from useful historical patterns vs. over-relying on misleading patterns. In this process of error balancing, you draw on the 10 commandments of superforecasting (Tetlock & Gardner, 2015) as well as on other peer-reviewed research on superforecasting:
Once you have written your reasons, ensure that they directly inform your forecast. Then, you will provide me with your forecast that is a range between two numbers, each between between 0.10 and 99.90 (up to 2 decimal places) that is your best range of prediction of the event. 
Output your prediction as “My Prediction: Between XX.XX% and YY.YY%, but ZZ.ZZ% being the most likely. Probability: ZZ.ZZ%." Please not add anything after. 

"""

import datetime
import json
import os
import requests
import re
from asknews_sdk import AskNewsSDK
from anthropic import Anthropic
from openai import OpenAI
import time
import logging
from asknews_sdk.errors import APIError

AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api2"
WARMUP_TOURNAMENT_ID = 3349

def find_number_before_percent(s):
    # Use a regular expression to find all numbers followed by a '%'
    matches = re.findall(r'(\d+(?:\.\d{1,2})?)%', s)
    if matches:
        # Return the last number found before a '%'
        return float(matches[-1])
    else:
        # Return None if no number found
        return None

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

def list_questions(tournament_id=WARMUP_TOURNAMENT_ID, offset=0, count=None):
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
    max_retries = 3
    base_delay = 1
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            if e.error_code == 500000:  # Internal Server Error
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
    """
    Use the AskNews `news` endpoint to get news context for your query.
    The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
    """
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
            n_articles=18,
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
  """
  Format the articles for posting to Metaculus.
  """

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

  return formatted_articles

#GPT-4 predictions
def get_gpt_prediction(question_details):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # client = OpenAI(api_key=OPENAI_API_KEY)

    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]

    formatted_articles = get_formatted_asknews_context(title)

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
                "content": PROMPT_TEMPLATE.format(
                    title=title,
                    formatted_articles=formatted_articles,
                    resolution_criteria=resolution_criteria,
                    today=today,
                    background=background,
                    fine_print=fine_print,
                )
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # This will raise an exception for HTTP errors
        response_data = response.json()
        gpt_text = response_data['choices'][0]['message']['content']
        return formatted_articles, gpt_text
    except requests.RequestException as e:
        print(f"Error in GPT prediction: {e}")
        return formatted_articles, None

def get_claude_prediction(question_details):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # client = Anthropic(api_key=ANTHROPIC_API_KEY)

    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]

    formatted_articles = get_formatted_asknews_context(title)

    url = "https://www.metaculus.com/proxy/anthropic/v1/messages"
    
    headers = {
        "Authorization": f"Token {METACULUS_TOKEN}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    title=title,
                    formatted_articles=formatted_articles,
                    resolution_criteria=resolution_criteria,
                    today=today,
                    background=background,
                    fine_print=fine_print,
                )
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # This will raise an exception for HTTP errors
        
        response_data = response.json()
        claude_text = response_data['content'][0]['text']
        return formatted_articles, claude_text
    except requests.RequestException as e:
            print(f"Error in Claude prediction: {e}")
            return formatted_articles, None

# Regular expression to find the number following 'Probability:
def extract_probability(ai_text):
    probability_match = find_number_before_percent(ai_text)

    # Extract the number if a match is found
    probability = None
    if probability_match:
        probability = float(probability_match) # int(match.group(1))
        print(f"The extracted probability is: {probability}%")
        probability = min(max(probability, 1), 99) # To prevent extreme forecasts
        return probability
    else:
        print("Unable to extract probability.")
        return None

#GPT prediction and submitting a forecast

questions = list_questions() #find open questions

open_questions_ids = []
for question in questions["results"]:
    if question["active_state"] == "OPEN":
        print(f"ID: {question['id']}\nQ: {question['title']}\nCloses: {question['close_time']}")
        open_questions_ids.append(question["id"])

SUBMIT_PREDICTION = True

for question_id in open_questions_ids:
    print(f"Question id: {question_id}\n\n")
    question_details = get_question_details(question_id)
    print("Question details:\n\n", question_details)

    formatted_articles, gpt_result = get_gpt_prediction(question_details)
    _, claude_result = get_claude_prediction(question_details)

    gpt_probability = extract_probability(gpt_result)
    claude_probability = extract_probability(claude_result)

    if gpt_probability is not None and claude_probability is not None:
        average_probability = (gpt_probability + claude_probability) / 2
        print(f"GPT probability: {gpt_probability}%")
        print(f"Claude probability: {claude_probability}%")
        print(f"Average probability: {average_probability}%")
        #print(f"GPT reasoning: {gpt_result}")
        #print(f"Claude reasoning: {claude_result}")

        if SUBMIT_PREDICTION:
            post_question_prediction(question_id, average_probability)
            comment = (f"Run 1\n\n" + gpt_result + "\n\n#########\n\n" + "Run 2\n\n" + claude_result)
            print(f"Posting comment: {comment}\n\n")
            post_question_comment(question_id, comment)
    else:
        print("Unable to extract probability.")