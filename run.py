import json
import os
import requests
import re
from scipy.interpolate import PchipInterpolator
from asknews_sdk import AskNewsSDK
import time
from asknews_sdk.errors import APIError
import sys
import logging
import datetime
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from prompts import BINARY_PROMPT, NUMERIC_PROMPT, MULTIPLE_CHOICE_PROMPT

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
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ASKNEWS_CLIENT_ID = os.environ.get('ASKNEWS_CLIENT_ID')
ASKNEWS_SECRET = os.environ.get('ASKNEWS_SECRET')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api"
TOURNAMENT_ID = 32506

# List questions and details

def setup_question_logger(post_id, log_type):
    """Set up a logger for a specific question and log type."""
    log_filename = f"logs/{post_id}_{log_type}.log"
    logger = logging.getLogger(f"{post_id}_{log_type}")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def log_question_reasoning(post_id, reasoning, question_title, model_name, run_number):
    """Log the reasoning for a specific question and run."""
    logger = setup_question_logger(post_id, "reasoning")
    logger.info(f"Question: {question_title}")
    logger.info(f"Reasoning for question {post_id}:\n{reasoning}")
    
    # JSON logging
    today = datetime.datetime.now().strftime('%Y%m%d')
    json_filename = f"logs/reasoning_{today}.json"
    
    question_data = {
        "question_id": post_id,
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
        existing_entry = next((item for item in existing_data if item["question_id"] == post_id), None)
        if existing_entry:
            existing_entry.update(question_data)
        else:
            existing_data.append(question_data)
        
        # Write updated data
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"Error writing to {json_filename}: {str(e)}")

def log_question_news(post_id, news, question_title):
    """Log the news articles for a specific question."""
    # Standard logging
    logger = setup_question_logger(post_id, "news")
    logger.info(f"Question: {question_title}")
    logger.info(f"News articles for question {post_id}:\n{news}")
    
    # JSON logging
    today = datetime.datetime.now().strftime('%Y%m%d')
    json_filename = f"logs/news_{today}.json"
    
    news_data = {
        "question_id": post_id,
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
        existing_entry = next((item for item in existing_data if item["question_id"] == post_id), None)
        if existing_entry:
            existing_entry.update(news_data)
        else:
            existing_data.append(news_data)
        
        # Write updated data
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"Error writing to {json_filename}: {str(e)}")

# Get questions from Metaculus
def list_questions(tournament_id=TOURNAMENT_ID, offset=0, count=None):
    """
    List open questions from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "has_group": "false",
        "order_by": "-activity",
        "forecast_type": ",".join([
            "binary",
            "multiple_choice",
            "numeric",
        ]),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    if count is not None:
        url_qparams["limit"] = count
    url = f"{API_BASE_URL}/posts/"
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
            premium=True,
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


#GPT-4 predictions
def get_binary_gpt_prediction(question_details, formatted_articles):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt_input = {
        "title": question_details["question"]["title"],
        "background": question_details["question"]["description"],
        "resolution_criteria": question_details["question"].get("resolution_criteria", ""),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today
    }
    
    max_retries = 10
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": BINARY_PROMPT.format(**prompt_input)
                }]
            )
            gpt_text = response.choices[0].message.content
            return gpt_text
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"GPT API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}")
                time.sleep(delay)
            else:
                logging.error(f"GPT API error persisted after {max_retries} retries: {e}")
                return None

def get_binary_claude_prediction(question_details, formatted_articles):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    prompt_input = {
        "title": question_details["question"]["title"],
        "background": question_details["question"]["description"],
        "resolution_criteria": question_details["question"].get("resolution_criteria", ""),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today
    }
    
    max_retries = 10
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": BINARY_PROMPT.format(**prompt_input)
                }]
            )
            claude_text = response.content[0].text
            return claude_text
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
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

def get_numeric_claude_prediction(question_details, formatted_articles):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt_input = {
        "title": question_details["question"]["title"],
        "background": question_details["question"]["description"],
        "resolution_criteria": question_details["question"].get("resolution_criteria", ""),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today,
        "lower_bound_message": f"The outcome can not be lower than {question_details['question']['scaling']['range_min']}." if not question_details["question"]["open_lower_bound"] else "",
        "upper_bound_message": f"The outcome can not be higher than {question_details['question']['scaling']['range_max']}." if not question_details["question"]["open_upper_bound"] else ""
    }

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    max_retries = 10
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": NUMERIC_PROMPT.format(**prompt_input)
                }]
            )
            claude_text = response.content[0].text
            
            percentile_values = extract_percentiles_from_response(claude_text)
            cdf = generate_continuous_cdf(
                percentile_values, 
                question_details["question"]["type"],
                question_details["question"]["open_upper_bound"],
                question_details["question"]["open_lower_bound"],
                question_details["question"]["scaling"]
            )
            
            comment = f"Extracted Percentile_values: {percentile_values}\n\nClaude's Answer: {claude_text}\n\n"
            return cdf, comment
            
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"Claude API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}")
                time.sleep(delay)
            else:
                logging.error(f"Claude API error persisted after {max_retries} retries: {e}")
                return None, None

def get_multiple_choice_claude_prediction(question_details, formatted_articles):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt_input = {
        "title": question_details["question"]["title"],
        "background": question_details["question"]["description"],
        "resolution_criteria": question_details["question"].get("resolution_criteria", ""),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today,
        "options": question_details["question"]["options"]
    }

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    max_retries = 10
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": MULTIPLE_CHOICE_PROMPT.format(**prompt_input)
                }]
            )
            claude_text = response.content[0].text
            
            option_probabilities = extract_option_probabilities_from_response(claude_text, question_details["question"]["options"])
            total_sum = sum(option_probabilities)
            decimal_list = [x / total_sum for x in option_probabilities]
            normalized_probabilities = normalize_list(decimal_list)
            
            probability_yes_per_category = {}
            for i, option in enumerate(question_details["question"]["options"]):
                probability_yes_per_category[option] = normalized_probabilities[i]
            
            comment = f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\nClaude's Answer: {claude_text}\n\n"
            return probability_yes_per_category, comment
            
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"Claude API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}")
                time.sleep(delay)
            else:
                logging.error(f"Claude API error persisted after {max_retries} retries: {e}")
                return None, None

def normalize_list(float_list):
    clamped_list = [max(min(x, 0.99), 0.01) for x in float_list]
    total_sum = sum(clamped_list)
    normalized_list = [x / total_sum for x in clamped_list]
    adjustment = 1.0 - sum(normalized_list)
    normalized_list[-1] += adjustment
    return normalized_list

def get_numeric_gpt_prediction(question_details, formatted_articles):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt_input = {
        "title": question_details["question"]["title"],
        "background": question_details["question"]["description"],
        "resolution_criteria": question_details["question"].get("resolution_criteria", ""),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today,
        "lower_bound_message": f"The outcome can not be lower than {question_details['question']['scaling']['range_min']}." if not question_details["question"]["open_lower_bound"] else "",
        "upper_bound_message": f"The outcome can not be higher than {question_details['question']['scaling']['range_max']}." if not question_details["question"]["open_upper_bound"] else ""
    }

    client = OpenAI(api_key=OPENAI_API_KEY)
    
    max_retries = 10
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": NUMERIC_PROMPT.format(**prompt_input)
                }]
            )
            gpt_text = response.choices[0].message.content
            
            percentile_values = extract_percentiles_from_response(gpt_text)
            cdf = generate_continuous_cdf(
                percentile_values, 
                question_details["question"]["type"],
                question_details["question"]["open_upper_bound"],
                question_details["question"]["open_lower_bound"],
                question_details["question"]["scaling"]
            )
            
            comment = f"Extracted Percentile_values: {percentile_values}\n\nGPT's Answer: {gpt_text}\n\n"
            return cdf, comment
            
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"GPT API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}")
                time.sleep(delay)
            else:
                logging.error(f"GPT API error persisted after {max_retries} retries: {e}")
                return None, None

def get_multiple_choice_gpt_prediction(question_details, formatted_articles):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt_input = {
        "title": question_details["question"]["title"],
        "background": question_details["question"]["description"],
        "resolution_criteria": question_details["question"].get("resolution_criteria", ""),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today,
        "options": question_details["question"]["options"]
    }

    client = OpenAI(api_key=OPENAI_API_KEY)
    
    max_retries = 10
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": MULTIPLE_CHOICE_PROMPT.format(**prompt_input)
                }]
            )
            gpt_text = response.choices[0].message.content
            
            option_probabilities = extract_option_probabilities_from_response(gpt_text, question_details["question"]["options"])
            total_sum = sum(option_probabilities)
            decimal_list = [x / total_sum for x in option_probabilities]
            normalized_probabilities = normalize_list(decimal_list)
            
            probability_yes_per_category = {}
            for i, option in enumerate(question_details["question"]["options"]):
                probability_yes_per_category[option] = normalized_probabilities[i]
            
            comment = f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\nGPT's Answer: {gpt_text}\n\n"
            return probability_yes_per_category, comment
            
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"GPT API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}")
                time.sleep(delay)
            else:
                logging.error(f"GPT API error persisted after {max_retries} retries: {e}")
                return None, None

def extract_percentiles_from_response(forecast_text: str) -> float:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_percentile_numbers(text):
        # Regular expression pattern
        pattern = r'^.*(?:P|p)ercentile.*$'

        # Number extraction pattern
        number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'

        results = []

        # Iterate through each line in the text
        for line in text.split('\n'):
            # Check if the line contains "Percentile" or "percentile"
            if re.match(pattern, line):
                # Extract all numbers from the line
                numbers = re.findall(number_pattern, line)
                numbers_no_commas = [num.replace(',', '') for num in numbers]
                # Convert strings to float or int
                numbers = [float(num) if '.' in num else int(num) for num in numbers_no_commas]
                # Add the tuple of numbers to results
                if len(numbers) > 1:
                  first_number = numbers[0]
                  last_number = numbers[-1]
                  tup = [first_number, last_number]
                  results.append(tuple(tup))

        # Convert results to dictionary
        percentile_values = {}
        for first_num, second_num in results:
            key = first_num
            percentile_values[key] = second_num

        return percentile_values

    percentile_values = extract_percentile_numbers(forecast_text)

    if len(percentile_values) > 0:
        return percentile_values
    else:
        raise ValueError(
            f"Could not extract prediction from response: {forecast_text}"
        )
    
def generate_continuous_cdf(
    percentile_values: dict,
    question_type: str,
    open_upper_bound: bool,
    open_lower_bound: bool,
    scaling: dict,
    use_monotonic_cubic: bool = True
) -> list[float]:
    range_min = float(scaling['range_min'])
    range_max = float(scaling['range_max'])

    def scale_to_unit(x):
        return (x - range_min) / (range_max - range_min)
    
    # Construct points array with lower and upper bounds
    points = []
    
    # Handle lower bound based on whether it's open or closed
    if open_lower_bound:
        points.append((0.0, 0.001))  # Small non-zero probability for open bound
    else:
        points.append((0.0, 0.0))    # Exactly zero probability for closed bound
    
    # Sort and deduplicate the percentile values
    sorted_percentiles = []
    seen_x_values = set()
    
    for pct, val in sorted(percentile_values.items()):
        val_clamped = max(min(val, range_max), range_min)
        x = scale_to_unit(val_clamped)
        
        # Add a small epsilon to x if we've seen this value before
        while x in seen_x_values:
            x += 1e-10
        
        seen_x_values.add(x)
        sorted_percentiles.append((x, pct / 100.0))

    # Add the sorted, deduplicated points
    points.extend(sorted_percentiles)

    # Handle upper bound based on whether it's open or closed
    if open_upper_bound:
        # Ensure the final point is strictly greater than the previous point
        last_x = points[-1][0]
        points.append((max(last_x + 1e-10, 1.0), 0.998))
    else:
        # For closed upper bound, ensure we reach exactly 1.0 probability
        points.append((1.0, 1.0))

    # Final sort to ensure x values are strictly increasing
    points.sort(key=lambda p: p[0])
    
    # Verify strict monotonicity
    for i in range(1, len(points)):
        if points[i][0] <= points[i-1][0]:
            points[i] = (points[i-1][0] + 1e-10, points[i][1])

    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]

    x_eval = np.linspace(0, 1, 201)

    if use_monotonic_cubic and len(points) >= 4:
        # Use monotonic cubic interpolation
        interpolator = PchipInterpolator(x_points, y_points)
        cdf_values = interpolator(x_eval)
    else:
        # Fall back to linear interpolation
        cdf_values = np.interp(x_eval, x_points, y_points)

    # Ensure exact bounds are respected
    cdf_values[0] = 0.0 if not open_lower_bound else 0.000
    cdf_values[-1] = 0.999 if not open_upper_bound else 0.999

    # Clip and enforce monotonicity and spacing
    cdf_values = np.clip(cdf_values, 0.0, 1.0)
    
    # Special handling to ensure proper spacing while respecting bounds
    for i in range(1, len(cdf_values)):
        if i == len(cdf_values) - 1 and not open_upper_bound:
            cdf_values[i] = 1.0  # Ensure last point is exactly 1.0 for closed upper bound
        else:
            cdf_values[i] = max(cdf_values[i], cdf_values[i-1] + 0.00005)

    return cdf_values.tolist()

def extract_option_probabilities_from_response(forecast_text, options):
    number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
    results = []

    for line in forecast_text.split('\n'):
        numbers = re.findall(number_pattern, line)
        numbers_no_commas = [num.replace(',', '') for num in numbers]
        numbers = [float(num) if '.' in num else int(num) for num in numbers_no_commas]
        if len(numbers) >= 1:
            last_number = numbers[-1]
            results.append(last_number)

    if len(results) > 0:
        return results[-len(options):]
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")

def get_summary_from_gpt(all_runs_text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    max_retries = 10
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model='gpt-4o',
                messages=[{
                    "role": "user",
                    "content": f"Please provide a concise summary of these forecasting runs, focusing on the key points of reasoning and how they led to the probabilities. You must include the probabilities from each run. Here are the runs:\n\n{all_runs_text}"
                }]
            )
            summary_text=response.choices[0].message.content
            return summary_text
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"GPT API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}")
                time.sleep(delay)
            else:
                logging.error(f"GPT API error persisted after {max_retries} retries: {e}")
                return None

def post_question_comment(post_id, comment_text):
    """
    Post a comment on the question page as the bot user.
    """
    response = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        **AUTH_HEADERS,
    )
    response.raise_for_status()

def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the api payload in the correct format.

    Args:
        forecast: The prediction value(s)
            - For binary: float probability (0-1)
            - For multiple choice: dict mapping options to probabilities
            - For numeric: list of 201 CDF values
        question_type: The type of question ("binary", "multiple_choice", or "numeric")
        
    Returns:
        dict: Properly formatted payload for the Metaculus API
    """
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric or date
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }

def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """
    Post a prediction on the question.
    
    Args:
        question_id: The Metaculus question ID (not post ID)
        forecast_payload: The prediction payload from create_forecast_payload()
    """
    url = f"{API_BASE_URL}/questions/forecast/"
    
    # API expects a list of forecasts
    response = requests.post(
        url,
        json=[
            {
                "question": question_id,
                **forecast_payload,
            },
        ],
        **AUTH_HEADERS,
    )
    
    logging.info(f"Response status: {response.status_code}")
    if not response.ok:
        logging.error(f"API Error: {response.status_code}")
        logging.error(f"Response content: {response.text}")
        raise Exception(response.text)
    
def log_predictions_json(post_id, question_title, gpt_results, claude_results, gpt_texts, claude_texts, average_probability):
    """Log predictions and reasoning to a JSON file."""
    today = datetime.datetime.now().strftime('%Y%m%d')
    json_filename = "logs/reasoning_{today}.json"
    
    prediction_data = {
        "question_id": post_id,
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
        existing_entry = next((item for item in existing_data if item["question_id"] == post_id), None)
        if existing_entry:
            existing_entry.update(prediction_data)
        else:
            existing_data.append(prediction_data)
            
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=2)
            
        logging.info(f"Successfully logged predictions for question {post_id} to {json_filename}")
    except Exception as e:
        logging.error(f"Error writing to {json_filename}: {str(e)}")

def get_question_details(post_id):
    """
    Get all details about a specific question.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(url)
    response = requests.get(
        url,
        **AUTH_HEADERS,
    )
    response.raise_for_status()
    return json.loads(response.content)

def generate_x_values(question_details):
    """Generate x-values based on question scaling"""
    scaling = question_details["question"]["scaling"]
    range_min = float(scaling.get("range_min"))
    range_max = float(scaling.get("range_max"))
    zero_point = scaling.get("zero_point")
    
    if zero_point is None:
        # Linear scale
        return np.linspace(range_min, range_max, 201)
    else:
        # Log scale
        deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
        return [range_min + (range_max - range_min) * (deriv_ratio**x - 1) / (deriv_ratio - 1) 
                for x in np.linspace(0, 1, 201)]

def combine_cdfs(cdf1, cdf2, x_values, open_upper_bound, open_lower_bound):
    """
    Combine two CDFs using a more robust method that preserves the distribution shape.
    
    Args:
        cdf1: First CDF array (from GPT-4)
        cdf2: Second CDF array (from Claude)
        x_values: The actual x-axis values these CDFs correspond to
        open_upper_bound: Whether the upper bound is open
        open_lower_bound: Whether the lower bound is open
        
    Returns:
        combined_cdf: Array of 201 points representing the combined CDF
    """
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    
    # Get percentiles from both CDFs
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    def get_value_at_percentile(cdf, x_vals, percentile):
        """Extract value at a given percentile from CDF"""
        target = percentile / 100.0
        # Find the first index where CDF exceeds target
        idx = np.searchsorted(cdf, target)
        if idx == 0:
            return x_vals[0]
        if idx == len(cdf):
            return x_vals[-1]
        # Interpolate between points
        x1, x2 = x_vals[idx-1], x_vals[idx]
        y1, y2 = cdf[idx-1], cdf[idx]
        return x1 + (x2 - x1) * (target - y1) / (y2 - y1)
    
    # Get values at percentiles for both CDFs
    values1 = [get_value_at_percentile(cdf1, x_values, p) for p in percentiles]
    values2 = [get_value_at_percentile(cdf2, x_values, p) for p in percentiles]
    
    # Average the values at each percentile
    combined_values = [(v1 + v2) / 2 for v1, v2 in zip(values1, values2)]
    
    # Create points for interpolation
    points = []
    
    # Handle lower bound
    if not open_lower_bound:
        points.append((x_values[0], 0.0))
    else:
        points.append((x_values[0], 0.001))
    
    # Add percentile points
    for p, v in zip(percentiles[1:-1], combined_values[1:-1]):
        points.append((v, p / 100.0))
    
    # Handle upper bound
    if not open_upper_bound:
        points.append((x_values[-1], 1.0))
    else:
        points.append((x_values[-1], 0.998))
    
    # Sort points and ensure strict monotonicity
    points.sort(key=lambda p: p[0])
    for i in range(1, len(points)):
        if points[i][0] <= points[i-1][0]:
            points[i] = (points[i-1][0] + 1e-10, points[i][1])
    
    # Create interpolator
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]
    
    if len(points) >= 4:
        interpolator = PchipInterpolator(x_points, y_points)
        combined_cdf = interpolator(x_values)
    else:
        # Fall back to linear interpolation if too few points
        combined_cdf = np.interp(x_values, x_points, y_points)
    
    # Ensure monotonicity and proper spacing
    combined_cdf = np.clip(combined_cdf, 0.0, 1.0)
    for i in range(1, len(combined_cdf)):
        if i == len(combined_cdf) - 1 and not open_upper_bound:
            combined_cdf[i] = 1.0
        else:
            combined_cdf[i] = max(combined_cdf[i], combined_cdf[i-1] + 0.00005)
    
    # Final validation of bounds
    if not open_lower_bound:
        combined_cdf[0] = 0.0
    if not open_upper_bound:
        combined_cdf[-1] = 1.0
        
    return combined_cdf.tolist()

def calculate_final_prediction(results, question_details):
    """Calculate final prediction based on question type"""
    question_type = results["type"]
    gpt_results = results["gpt_results"]
    claude_results = results["claude_results"]
    
    if question_type == "binary":
        gpt_avg = sum(gpt_results) / len(gpt_results)
        claude_avg = sum(claude_results) / len(claude_results)
        final_prediction = (gpt_avg + claude_avg) / 2
        return {"prediction": final_prediction / 100}
        
    elif question_type == "numeric":
        # Get the actual x-values based on question range
        x_values = generate_x_values(question_details)
        
        # Average across runs first
        gpt_avg_cdf = np.mean(gpt_results, axis=0)
        claude_avg_cdf = np.mean(claude_results, axis=0)
        
        # Combine the averaged CDFs
        final_prediction = combine_cdfs(gpt_avg_cdf, claude_avg_cdf, x_values, question_details["question"]["open_upper_bound"], question_details["question"]["open_lower_bound"])
        
        # The API expects exactly 201 points
        assert len(final_prediction) == 201, f"CDF must have 201 points, got {len(final_prediction)}"
        
        # Each point should be between 0 and 1
        final_prediction = [min(max(p, 0.0), 1.0) for p in final_prediction]
        
        return {"continuous_cdf": final_prediction}
        
    elif question_type == "multiple_choice":
        # Multiple choice handling as before
        combined_probs = {}
        all_options = set().union(*[r.keys() for r in gpt_results + claude_results])
        
        for option in all_options:
            gpt_option_probs = [result.get(option, 0) for result in gpt_results]
            claude_option_probs = [result.get(option, 0) for result in claude_results]
            all_probs = gpt_option_probs + claude_option_probs
            combined_probs[option] = sum(all_probs) / len(all_probs)
            
        return {"prediction_probs": combined_probs}
    
    return None

SUBMIT_PREDICTION = True

#Submitting a forecast
def main():
    """Main function to process questions and submit forecasts."""
    posts = list_questions()
    
    # Create mapping of post IDs to questions
    post_dict = {}
    for post in posts["results"]:
        if question := post.get("question"):
            post_dict[post["id"]] = [question]
            logging.info(f'Post ID: {post["id"]}, Question ID: {post["question"]["id"]}')

    # Get list of open questions with both post and question IDs
    open_question_id_post_id = []  # [(question_id, post_id)]
    print(open_question_id_post_id)
    
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                logging.info(
                    f"ID: {question['id']}\nQ: {question['title']}\n"
                    f"Closes: {question['scheduled_close_time']}"
                )
                open_question_id_post_id.append((question["id"], post_id))

    # Process each open question
    for question_id, post_id in open_question_id_post_id:
        try:
            logging.info(f"\nProcessing question ID: {question_id}, post ID: {post_id}")
            
            # Get question details using the post_id
            question_details = get_question_details(post_id)
            question_type = question_details["question"]["type"]
            
            # Get news context
            formatted_articles = get_formatted_asknews_context(question_details["title"])
            log_question_news(post_id, formatted_articles, question_details["question"]["title"])
            
            # Initialize results storage
            gpt_results = []
            claude_results = []
            gpt_texts = []
            claude_texts = []
            
            # Number of runs based on question type
            num_runs = 5 if question_type == "binary" else 1
            
            # Process predictions for each run
            for run in range(num_runs):
                logging.info(f"Run {run + 1}/{num_runs}")
                
                if question_type == "binary":
                    # Binary predictions
                    gpt_result = get_binary_gpt_prediction(question_details, formatted_articles)
                    claude_result = get_binary_claude_prediction(question_details, formatted_articles)
                    
                    if gpt_result:
                        gpt_prob = extract_probability(gpt_result)
                        if gpt_prob is not None:
                            gpt_results.append(gpt_prob)
                        gpt_texts.append(gpt_result)
                    
                    if claude_result:
                        claude_prob = extract_probability(claude_result)
                        if claude_prob is not None:
                            claude_results.append(claude_prob)
                        claude_texts.append(claude_result)
                
                elif question_type == "numeric":
                    # Numeric predictions
                    gpt_result, gpt_comment = get_numeric_gpt_prediction(question_details, formatted_articles)
                    claude_result, claude_comment = get_numeric_claude_prediction(question_details, formatted_articles)
                    
                    if gpt_result and claude_result:
                        gpt_results.append(gpt_result)
                        claude_results.append(claude_result)
                        gpt_texts.append(gpt_comment)
                        claude_texts.append(claude_comment)
                
                elif question_type == "multiple_choice":
                    # Multiple choice predictions
                    gpt_result, gpt_comment = get_multiple_choice_gpt_prediction(question_details, formatted_articles)
                    claude_result, claude_comment = get_multiple_choice_claude_prediction(question_details, formatted_articles)
                    
                    if gpt_result and claude_result:
                        gpt_results.append(gpt_result)
                        claude_results.append(claude_result)
                        gpt_texts.append(gpt_comment)
                        claude_texts.append(claude_comment)
                
                # Log individual model results
                if gpt_result:
                    log_question_reasoning(post_id, gpt_texts[-1], question_details["question"]["title"], "gpt", run)
                if claude_result:
                    log_question_reasoning(post_id, claude_texts[-1], question_details["question"]["title"], "claude", run)

            # Process and submit final prediction if we have results from both models
            if gpt_results and claude_results:
                # Calculate final prediction
                results = {
                    "type": question_type,
                    "gpt_results": gpt_results,
                    "claude_results": claude_results
                }
                final_prediction = calculate_final_prediction(results, question_details)
                
                if final_prediction:
                    # Create the API payload
                    forecast_payload = create_forecast_payload(
                        final_prediction["prediction" if question_type == "binary" else 
                                      "prediction_probs" if question_type == "multiple_choice" else 
                                      "continuous_cdf"],
                        question_type
                    )
                    
                    # Get prediction summary
                    summary = get_summary_from_gpt(
                        f"Analyze these forecasting runs for a {question_type} question:\n\n"
                        f"GPT analysis:\n{gpt_texts[-1]}\n\n"
                        f"Claude analysis:\n{claude_texts[-1]}\n\n"
                    )
                    
                    # Log predictions
                    log_predictions_json(
                        post_id,
                        question_details["question"]["title"],
                        gpt_results,
                        claude_results,
                        gpt_texts,
                        claude_texts,
                        final_prediction
                    )
                    
                    if SUBMIT_PREDICTION:
                        # Submit prediction using question_id
                        post_question_prediction(question_id, forecast_payload)
                        
                        # Format and post comment based on question type
                        if question_type == "binary":
                            pred_value = final_prediction["prediction"] * 100
                            comment = f"Summary of {num_runs} runs:\n\n{summary}\n\nFinal prediction: {pred_value:.2f}%"
                        elif question_type == "numeric":
                            comment = (
                                f"Summary of prediction:\n\n{summary}\n\n"
                                f"Numeric prediction submitted (CDF with 201 points)\n"
                                f"Key percentiles:\n"
                                f"- 25th: {np.percentile(final_prediction['continuous_cdf'], 25):.2f}\n"
                                f"- 50th: {np.percentile(final_prediction['continuous_cdf'], 50):.2f}\n"
                                f"- 75th: {np.percentile(final_prediction['continuous_cdf'], 75):.2f}"
                            )
                        else:  # multiple_choice
                            probs = final_prediction["prediction_probs"]
                            probs_text = "\n".join([f"{option}: {prob*100:.2f}%" for option, prob in probs.items()])
                            comment = f"Summary of prediction:\n\n{summary}\n\nFinal predictions:\n{probs_text}"
                        
                        # Post comment using post_id
                        post_question_comment(post_id, comment)
                        logging.info(f"Successfully submitted prediction for question {post_id}")
            
            else:
                logging.warning(f"No valid predictions generated for question {post_id}")

        except Exception as e:
            logging.error(f"Error processing question {post_id}: {str(e)}")
            continue
        
        # Sleep to avoid rate limiting
        time.sleep(2)

if __name__ == "__main__":
    main()
