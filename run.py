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
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/annabot_newsonnet.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

args = sys.argv
if len(args) > 1 and args[1] == "dryrun":
    print("dry run, doing nothing")
    exit(0)

METACULUS_TOKEN = os.environ.get("METACULUS_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ASKNEWS_CLIENT_ID = os.environ.get("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.environ.get("ASKNEWS_SECRET")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api"
TOURNAMENT_ID = 32627
ANTHROPIC_PROXY = {
    "url": "https://www.metaculus.com/proxy/anthropic/v1/messages/",
    "headers": {
        "Authorization": f"Token {METACULUS_TOKEN}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
}
OPENAI_PROXY = {
    "url": "https://www.metaculus.com/proxy/openai/v1/chat/completions/",
    "headers": {
        "Authorization": f"Token {METACULUS_TOKEN}",
        "Content-Type": "application/json",
    }
}

# List questions and details

def setup_question_logger(post_id, log_type):
    """Set up a logger for a specific question and log type."""
    log_filename = f"logs/{post_id}_{log_type}.log"
    logger = logging.getLogger(f"{post_id}_{log_type}")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def update_json_log(filename, data, post_id):
    """
    Generic function to update JSON log files.
    Returns True if successful, False otherwise.
    """
    try:
        # Read existing data if file exists
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = []

        # Update existing entry or add new one
        existing_entry = next(
            (item for item in existing_data if item["question_id"] == post_id), None
        )
        if existing_entry:
            existing_entry.update(data)
        else:
            existing_data.append(data)

        # Write updated data
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=2)
        return True

    except Exception as e:
        logging.error(f"Error writing to {filename}: {str(e)}")
        return False


def log_question_reasoning(post_id, reasoning, question_title, model_name, run_number):
    """Log the reasoning for a specific question and run."""
    # Standard logging
    logger = setup_question_logger(post_id, "reasoning")
    logger.info(f"Question: {question_title}")
    logger.info(f"Reasoning for question {post_id}:\n{reasoning}")

    # JSON logging
    today = datetime.datetime.now().strftime("%Y%m%d")
    json_filename = f"logs/reasoning_{today}.json"

    question_data = {
        "question_id": post_id,
        "question_title": question_title,
        f"{model_name}_reasoning{run_number}": reasoning,
    }

    update_json_log(json_filename, question_data, post_id)


def log_predictions_json(
    post_id,
    question_title,
    gpt_results,
    claude_results,
    gpt_texts,
    claude_texts,
    average_probability,
):
    """Log predictions and reasoning to a JSON file."""
    today = datetime.datetime.now().strftime("%Y%m%d")
    json_filename = f"logs/reasoning_{today}.json"

    prediction_data = {
        "question_id": post_id,
        "question_title": question_title,
        "timestamp": datetime.datetime.now().isoformat(),
        "runs": [],
        "average_probability": average_probability,
    }

    for i in range(len(gpt_results)):
        prediction_data["runs"].append(
            {
                "run_number": i + 1,
                "gpt_prediction": gpt_results[i],
                "gpt_reasoning": gpt_texts[i],
                "claude_prediction": claude_results[i],
                "claude_reasoning": claude_texts[i],
            }
        )

    success = update_json_log(json_filename, prediction_data, post_id)
    if success:
        logging.info(
            f"Successfully logged predictions for question {post_id} to {json_filename}"
        )


def log_question_news(post_id, news, question_title):
    """Log the news articles for a specific question."""
    # Standard logging
    logger = setup_question_logger(post_id, "news")
    logger.info(f"Question: {question_title}")
    logger.info(f"News articles for question {post_id}:\n{news}")

    # JSON logging
    today = datetime.datetime.now().strftime("%Y%m%d")
    json_filename = f"logs/news_{today}.json"

    news_data = {"question_id": post_id, "question_title": question_title, "news": news}
    update_json_log(json_filename, news_data, post_id)


# Get questions from Metaculus
def list_questions(tournament_id=TOURNAMENT_ID, offset=0, count=100):
    """
    List open questions from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "has_group": "false",
        "order_by": "-activity",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
            ]
        ),
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
                    delay = base_delay * (2**attempt)
                    logging.warning(
                        f"AskNews API Internal Server Error. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logging.error(
                        "AskNews API Internal Server Error persisted after max retries."
                    )
                    raise
            else:
                raise


def get_formatted_asknews_context(query):
    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes=["news"]
    )

    try:
        # get the latest news related to the query (within the past 48 hours)
        hot_response = asknews_api_call_with_retry(
            ask.news.search_news,
            query=query,  # your natural language query
            n_articles=5,  # control the number of articles to include in the context
            return_type="both",
            premium=True,
            strategy="latest news",  # enforces looking at the latest news only
        )

        # get context from the "historical" database that contains a news archive going back to 2023
        historical_response = asknews_api_call_with_retry(
            ask.news.search_news,
            query=query,
            n_articles=25,
            return_type="both",
            strategy="news knowledge",  # looks for relevant news within the past 60 days
        )

        formatted_articles = format_asknews_context(
            hot_response.as_dicts, historical_response.as_dicts
        )
    except APIError as e:
        logging.error(f"AskNews API error: {e}")
        formatted_articles = "Error fetching news articles. Please try again later."

    return formatted_articles


def format_asknews_context(hot_articles, historical_articles):
    formatted_articles = "Here are the relevant news articles:\n\n"

    if hot_articles:
        hot_articles = [article.__dict__ for article in hot_articles]
        hot_articles = sorted(hot_articles, key=lambda x: x["pub_date"], reverse=True)

        for article in hot_articles:
            pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
            formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if historical_articles:
        historical_articles = [article.__dict__ for article in historical_articles]
        historical_articles = sorted(
            historical_articles, key=lambda x: x["pub_date"], reverse=True
        )

        for article in historical_articles:
            pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
            formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if not hot_articles and not historical_articles:
        formatted_articles += "No articles were found.\n\n"
        return formatted_articles

    formatted_articles += f"*Generated by AI at [AskNews](https://asknews.app), check out the [API](https://docs.asknews.app) for more information*."
    logging.info(f"News articles:\n{formatted_articles}")
    return formatted_articles


# GPT-4 predictions
def get_binary_gpt_prediction(question_details, formatted_articles):
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    prompt_input = {
        "title": question_details["question"]["title"],
        "background": question_details["question"]["description"],
        "resolution_criteria": question_details["question"].get(
            "resolution_criteria", ""
        ),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today,
    }

    url = OPENAI_PROXY["url"]
    headers = OPENAI_PROXY["headers"]

    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": BINARY_PROMPT.format(**prompt_input)}],
    }

    max_retries = 10
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            gpt_text = response_data["choices"][0]["message"]["content"]
            return gpt_text
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logging.warning(
                    f"OpenAI API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}"
                )
                time.sleep(delay)
            else:
                logging.error(
                    f"OpenAI API error persisted after {max_retries} retries: {e}"
                )
                return None


def get_binary_claude_prediction(question_details, formatted_articles):
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    prompt_input = {
        "title": question_details["question"]["title"],
        "background": question_details["question"]["description"],
        "resolution_criteria": question_details["question"].get(
            "resolution_criteria", ""
        ),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today,
    }

    url = ANTHROPIC_PROXY["url"]
    headers = ANTHROPIC_PROXY["headers"]

    data = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": BINARY_PROMPT.format(**prompt_input)}],
    }

    max_retries = 10
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()

            response_data = response.json()
            claude_text = response_data["content"][0]["text"]
            return claude_text
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logging.warning(
                    f"Anthropic API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}"
                )
                time.sleep(delay)
            else:
                logging.error(
                    f"Anthropic API error persisted after {max_retries} retries: {e}"
                )
                return None


# Find all numbers followed by a '%'
def find_number_before_percent(s):
    matches = re.findall(r"(\d+(?:\.\d{1,2})?)%", s)
    if matches:
        return float(matches[-1])
    else:
        return None


def extract_probability(ai_text):
    probability_match = find_number_before_percent(ai_text)

    # Extract the number if a match is found
    probability = None
    if probability_match:
        probability = float(probability_match)  # int(match.group(1))
        logging.info(f"The extracted probability is: {probability}%")
        probability = min(max(probability, 1), 99)  # To prevent extreme forecasts
        return probability
    else:
        print("Unable to extract probability.")
        return None


def get_numeric_claude_prediction(question_details, formatted_articles):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt_input = {
        "title": question_details["question"]["title"],
        "background": question_details["question"]["description"],
        "resolution_criteria": question_details["question"].get(
            "resolution_criteria", ""
        ),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today,
        "lower_bound_message": (
            f"The outcome can not be lower than {question_details['question']['scaling']['range_min']}."
            if not question_details["question"]["open_lower_bound"]
            else ""
        ),
        "upper_bound_message": (
            f"The outcome can not be higher than {question_details['question']['scaling']['range_max']}."
            if not question_details["question"]["open_upper_bound"]
            else ""
        ),
    }

    url = ANTHROPIC_PROXY["url"]
    headers = ANTHROPIC_PROXY["headers"]

    data = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": NUMERIC_PROMPT.format(**prompt_input)}
        ],
    }

    max_retries = 10
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            claude_text = response_data["content"][0]["text"]

            percentile_values = extract_percentiles_from_response(claude_text)
            cdf = generate_continuous_cdf(
                percentile_values,
                question_details["question"]["type"],
                question_details["question"]["open_upper_bound"],
                question_details["question"]["open_lower_bound"],
                question_details["question"]["scaling"],
            )

            comment = f"Extracted Percentile_values: {percentile_values}\n\nClaude's Answer: {claude_text}\n\n"
            return cdf, comment

        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logging.warning(
                    f"Anthropic API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}"
                )
                time.sleep(delay)
            else:
                logging.error(
                    f"Anthropic API error persisted after {max_retries} retries: {e}"
                )
                return None, None


def get_multiple_choice_claude_prediction(question_details, formatted_articles):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt_input = {
        "title": question_details["question"]["title"],
        "background": question_details["question"]["description"],
        "resolution_criteria": question_details["question"].get(
            "resolution_criteria", ""
        ),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today,
        "options": question_details["question"]["options"],
    }

    url = ANTHROPIC_PROXY["url"]
    headers = ANTHROPIC_PROXY["headers"]

    data = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": MULTIPLE_CHOICE_PROMPT.format(**prompt_input)}
        ],
    }

    max_retries = 10
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            claude_text = response_data["content"][0]["text"]

            option_probabilities = extract_option_probabilities_from_response(
                claude_text, question_details["question"]["options"]
            )
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
                delay = base_delay * (2**attempt)
                logging.warning(
                    f"Anthropic API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}"
                )
                time.sleep(delay)
            else:
                logging.error(
                    f"Anthropic API error persisted after {max_retries} retries: {e}"
                )
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
        "resolution_criteria": question_details["question"].get(
            "resolution_criteria", ""
        ),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today,
        "lower_bound_message": (
            f"The outcome can not be lower than {question_details['question']['scaling']['range_min']}."
            if not question_details["question"]["open_lower_bound"]
            else ""
        ),
        "upper_bound_message": (
            f"The outcome can not be higher than {question_details['question']['scaling']['range_max']}."
            if not question_details["question"]["open_upper_bound"]
            else ""
        ),
    }

    url = OPENAI_PROXY["url"]
    headers = OPENAI_PROXY["headers"]

    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": NUMERIC_PROMPT.format(**prompt_input)}
        ],
    }

    max_retries = 10
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            gpt_text = response_data["choices"][0]["message"]["content"]

            percentile_values = extract_percentiles_from_response(gpt_text)
            cdf = generate_continuous_cdf(
                percentile_values,
                question_details["question"]["type"],
                question_details["question"]["open_upper_bound"],
                question_details["question"]["open_lower_bound"],
                question_details["question"]["scaling"],
            )

            comment = f"Extracted Percentile_values: {percentile_values}\n\nGPT's Answer: {gpt_text}\n\n"
            return cdf, comment

        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logging.warning(
                    f"OpenAI API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}"
                )
                time.sleep(delay)
            else:
                logging.error(
                    f"OpenAI API error persisted after {max_retries} retries: {e}"
                )
                return None, None


def get_multiple_choice_gpt_prediction(question_details, formatted_articles):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt_input = {
        "title": question_details["question"]["title"],
        "background": question_details["question"]["description"],
        "resolution_criteria": question_details["question"].get(
            "resolution_criteria", ""
        ),
        "fine_print": question_details["question"].get("fine_print", ""),
        "formatted_articles": formatted_articles,
        "today": today,
        "options": question_details["question"]["options"],
    }

    url = OPENAI_PROXY["url"]
    headers = OPENAI_PROXY["headers"]

    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": MULTIPLE_CHOICE_PROMPT.format(**prompt_input)}
        ],
    }

    max_retries = 10
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            gpt_text = response_data["choices"][0]["message"]["content"]

            option_probabilities = extract_option_probabilities_from_response(
                gpt_text, question_details["question"]["options"]
            )
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
                delay = base_delay * (2**attempt)
                logging.warning(
                    f"OpenAI API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}"
                )
                logging.error(
                    f"Response content: {e.response.text if hasattr(e, 'response') else 'No response'}"
                )
                time.sleep(delay)
            else:
                logging.error(
                    f"OpenAI API error persisted after {max_retries} retries: {e}"
                )
                return None, None


def extract_percentiles_from_response(forecast_text: str) -> float:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_percentile_numbers(text):
        # Regular expression pattern
        pattern = r"^.*(?:P|p)ercentile.*$"

        # Number extraction pattern
        number_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?"

        results = []

        # Iterate through each line in the text
        for line in text.split("\n"):
            # Check if the line contains "Percentile" or "percentile"
            if re.match(pattern, line):
                # Extract all numbers from the line
                numbers = re.findall(number_pattern, line)
                numbers_no_commas = [num.replace(",", "") for num in numbers]
                # Convert strings to float or int
                numbers = [
                    float(num) if "." in num else int(num) for num in numbers_no_commas
                ]
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
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def generate_continuous_cdf(
    percentile_values: dict,
    question_type: str,
    open_upper_bound: bool,
    open_lower_bound: bool,
    scaling: dict,
    use_monotonic_cubic: bool = True,
) -> list[float]:
    """
    Generate a strictly monotonic CDF for numeric (and date) questions on Metaculus,
    respecting open/closed bounds at 0/1 or 0.001/0.999, a minimum step of 1e-5,
    and a maximum step of 0.59 between consecutive points.

    Args:
        percentile_values: Dict of {percentile -> raw_value}, e.g. {25: 10.0, 50: 15.0, ...}
        question_type: Metaculus question type (usually 'numeric' or 'date' for continuous CDF).
        open_upper_bound: If True, upper bound is open (CDF ~ 0.999); else it is closed (CDF ~ 1.0).
        open_lower_bound: If True, lower bound is open (CDF ~ 0.001); else it is closed (CDF ~ 0.0).
        scaling: Dict containing 'range_min' and 'range_max' for linear scaling. e.g. {"range_min": 0, "range_max": 100}.
        use_monotonic_cubic: Whether to use a PCHIP monotonic spline if sufficient anchor points exist.

    Returns:
        A list of length 201, representing the continuous CDF in 201 steps from x=0 to x=1,
        strictly monotonic, respecting step-size constraints, and matching open/closed bounds.
    """

    range_min = float(scaling["range_min"])
    range_max = float(scaling["range_max"])

    def scale_to_unit(x):
        """Scale raw numeric question value x to [0..1]."""
        return (x - range_min) / (range_max - range_min)

    # Construct points array with lower and upper bounds
    points = []

    if open_lower_bound:
        # Start at (x=0.0, probability=0.001)
        points.append((0.0, 0.001))
    else:
        # Start at (x=0.0, probability=0.0)
        points.append((0.0, 0.0))

    # Convert user’s percentile values to x ∈ [0..1], p ∈ [0..1].
    # Sort by percentile in ascending order (e.g. 5th -> 50th -> 95th).
    sorted_percentiles = sorted(percentile_values.items(), key=lambda kv: kv[0])

    # Keep track of x-values to avoid duplicates
    seen_x_values = set([points[0][0]])

    for pct, raw_val in sorted_percentiles:
        # Clamp the raw value to [range_min, range_max] so we don't go out of range
        clamped_val = max(min(raw_val, range_max), range_min)
        x = scale_to_unit(clamped_val)
        # Convert percentile to fraction in [0..1]
        p = pct / 100.0

        while x in seen_x_values:
            seen_x_values.add(x)
        points.append((x, p))

    # Upper bound
    if open_upper_bound:
        # Add final open-bound at x=1.0 with probability=0.999
        # but ensure x is strictly greater than last anchor
        last_x = points[-1][0]
        if last_x >= 1.0:
            # In rare case we have an anchor at 1.0 already
            last_x = 1.0 - 1e-5
        points.append((max(last_x + 1e-5, 1.0), 0.999))
    else:
        # For closed upper bound, ensure we reach exactly 1.0 probability
        points.append((1.0, 1.0))

    # Final sort to ensure x values are strictly increasing
    points.sort(key=lambda p: p[0])

    # Verify strict monotonicity
    for i in range(1, len(points)):
        if points[i][0] <= points[i - 1][0]:
            points[i] = (points[i - 1][0] + 5e-5, points[i][1])

    x_points = [pt[0] for pt in points]
    y_points = [pt[1] for pt in points]

    x_eval = np.linspace(0, 1, 201)

    if use_monotonic_cubic and len(points) >= 4:
        # Use monotonic cubic interpolation
        interpolator = PchipInterpolator(x_points, y_points)
        cdf_values = interpolator(x_eval)
    else:
        # Fall back to linear interpolation
        cdf_values = np.interp(x_eval, x_points, y_points)

    if open_lower_bound:
        cdf_values[0] = max(cdf_values[0], 0.001)
    else:
        cdf_values[0] = 0.0

    # Clip and enforce monotonicity and spacing
    if open_upper_bound:
        cdf_values[-1] = min(cdf_values[-1], 0.999)
    else:
        cdf_values[-1] = 1.0

    # Clip to [0,1] just in case
    cdf_values = np.clip(cdf_values, 0.0, 1.0)

    # Special handling to ensure proper spacing while respecting bounds
    # Left-to-right pass
    for i in range(1, len(cdf_values)):
        # Must be >= the previous + 1e-5
        min_allowed = cdf_values[i - 1] + 1e-5
        # Must be <= the previous + 0.59
        max_allowed = cdf_values[i - 1] + 0.59
        cdf_values[i] = np.clip(cdf_values[i], min_allowed, max_allowed)

    # Right-to-left pass, in case large jumps are forced from the right side
    for i in range(len(cdf_values) - 2, -1, -1):
        # cdf_values[i] must be <= cdf_values[i+1] - 1e-5 Actually we want cdf[i] < cdf[i+1], so:
        max_allowed = cdf_values[i + 1] - 1e-5
        # cdf_values[i] must be >= cdf_values[i+1] - 0.59
        min_allowed = cdf_values[i + 1] - 0.59
        cdf_values[i] = np.clip(cdf_values[i], min_allowed, max_allowed)

    # Final clip for safety
    cdf_values = np.clip(cdf_values, 0.0, 1.0)

    # Now re-enforce the exact endpoints one last time
    if open_lower_bound:
        cdf_values[0] = max(cdf_values[0], 0.001)
    else:
        cdf_values[0] = 0.0
    if open_upper_bound:
        cdf_values[-1] = min(cdf_values[-1], 0.999)
    else:
        cdf_values[-1] = 1.0

    return cdf_values.tolist()

def extract_option_probabilities_from_response(forecast_text, options):
    number_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
    results = []

    for line in forecast_text.split("\n"):
        numbers = re.findall(number_pattern, line)
        numbers_no_commas = [num.replace(",", "") for num in numbers]
        numbers = [float(num) if "." in num else int(num) for num in numbers_no_commas]
        if len(numbers) >= 1:
            last_number = numbers[-1]
            results.append(last_number)

    if len(results) > 0:
        return results[-len(options) :]
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def get_summary_from_gpt(all_runs_text):
    max_retries = 10
    base_delay = 1

    url = OPENAI_PROXY["url"]
    headers = OPENAI_PROXY["headers"]

    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": f"Please provide a concise summary of these forecasting runs, focusing on the key points of reasoning and how they led to the probabilities. You must include the probabilities from each run. Here are the runs:\n\n{all_runs_text}",
            }
        ],
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        except requests.RequestExceptionq as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logging.warning(
                    f"OpenAI API error on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds... Error: {e}"
                )
                time.sleep(delay)
            else:
                logging.error(
                    f"OpenAI API error persisted after {max_retries} retries: {e}"
                )
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


def get_question_details(post_id):
    """
    Get all details about a specific question and verify no forecast exists.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    
    response = requests.get(url, **AUTH_HEADERS)
    response.raise_for_status()
    details = json.loads(response.content)

    # Check if question exists in response
    if "question" not in details:
        return None
        
    my_forecasts = details["question"].get("my_forecasts", {})
    
    # Check if there's no existing forecast
    if my_forecasts.get("latest") is None:
        return details
    
    return None


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
        return [
            range_min
            + (range_max - range_min) * (deriv_ratio**x - 1) / (deriv_ratio - 1)
            for x in np.linspace(0, 1, 201)
        ]


def combine_cdfs(cdf1, cdf2, x_values, open_upper_bound, open_lower_bound):
    """
    Combine two CDFs while respecting bounds constraints.

    Args:
        cdf1, cdf2: Lists of 201 probability values (the CDFs).
        x_values:   The corresponding x-values (length 201).
        open_upper_bound:  Boolean; if True, the final CDF must be <= 0.999.
        open_lower_bound:  Boolean; if True, the initial CDF must be >= 0.001.

    Returns:
        A new combined CDF (list of 201 floats) that:
        - Is strictly increasing by at least 5e-05 at each step
        - Respects open/closed bounds at start (0.0 or 0.001) and end (1.0 or 0.999).
    """

    # Step 1: Convert each CDF -> "x-values at each percentile"
    percentiles = np.linspace(0, 100, 201)
    
    def get_value_at_percentile(cdf, x_vals, pct):
        """Find x-value where cdf ~ pct%."""
        frac = pct / 100.0
        if frac <= 0:
            return x_vals[0]
        if frac >= 1:
            return x_vals[-1]
        idx = np.searchsorted(cdf, frac)
        if idx == 0:
            return x_vals[0]
        if idx >= len(cdf):
            return x_vals[-1]
        x1, x2 = x_vals[idx - 1], x_vals[idx]
        y1, y2 = cdf[idx - 1], cdf[idx]
        if abs(y2 - y1) < 1e-12:
            return x1
        return x1 + (x2 - x1) * (frac - y1) / (y2 - y1)
    
    values1 = [get_value_at_percentile(cdf1, x_values, p) for p in percentiles]
    values2 = [get_value_at_percentile(cdf2, x_values, p) for p in percentiles]
    avg_vals = [(v1 + v2) / 2 for v1, v2 in zip(values1, values2)]
    
    # Step 2: Convert back to a CDF on [0..1] by interpolation
    def get_probability_from_value(val, avg_xvals, pcts):
        """Given val in x-axis, find the cdf fraction."""
        if val <= avg_xvals[0]:
            return 0.0 if not open_lower_bound else 0.001
        if val >= avg_xvals[-1]:
            return 1.0 if not open_upper_bound else 0.999
        idx = np.searchsorted(avg_xvals, val)
        if idx == 0:
            return 0.0 if not open_lower_bound else 0.001
        if idx >= len(avg_xvals):
            return 1.0 if not open_upper_bound else 0.999
        x1, x2 = avg_xvals[idx - 1], avg_xvals[idx]
        f1, f2 = pcts[idx - 1] / 100.0, pcts[idx] / 100.0
        if abs(x2 - x1) < 1e-12:
            return f1
        return f1 + (f2 - f1) * (val - x1) / (x2 - x1)
    
    combined_cdf = [get_probability_from_value(x, avg_vals, percentiles) for x in x_values]

    # Step 3: Force the first and last points according to open/closed
    if open_lower_bound:
        combined_cdf[0] = max(combined_cdf[0], 0.001)
    else:
        combined_cdf[0] = 0.0
    if open_upper_bound:
        combined_cdf[-1] = min(combined_cdf[-1], 0.999)
    else:
        combined_cdf[-1] = 1.0

    # Step 4: Enforce strict increments of 5e-05 (left-to-right)
    for i in range(1, len(combined_cdf)):
        combined_cdf[i] = max(combined_cdf[i], combined_cdf[i - 1] + 5e-05)

    # Step 5: Enforce the upper bound again if open
    if open_upper_bound:
        combined_cdf[-1] = min(combined_cdf[-1], 0.999)
    
    # Step 6: Right-to-left pass to avoid overshoots
    for i in range(len(combined_cdf) - 2, -1, -1):
        combined_cdf[i] = min(combined_cdf[i], combined_cdf[i + 1] - 5e-05)

    # Step 7: Final clamp
    combined_cdf = np.clip(combined_cdf, 0.0, 1.0)
    
    # Step 8: Re-assert the very first and last in case the passes shifted them
    if open_lower_bound:
        combined_cdf[0] = max(combined_cdf[0], 0.001)
    else:
        combined_cdf[0] = 0.0
    if open_upper_bound:
        combined_cdf[-1] = min(combined_cdf[-1], 0.999)
    else:
        combined_cdf[-1] = 1.0

    # Step 9: Final pass to fix any new violation
    for i in range(1, len(combined_cdf)):
        combined_cdf[i] = max(combined_cdf[i], combined_cdf[i - 1] + 5e-05)
    combined_cdf = np.clip(combined_cdf, 0.0, 1.0)

    return combined_cdf


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
        final_prediction = combine_cdfs(
            gpt_avg_cdf,
            claude_avg_cdf,
            x_values,
            question_details["question"]["open_upper_bound"],
            question_details["question"]["open_lower_bound"],
        )

        # The API expects exactly 201 points
        assert (
            len(final_prediction) == 201
        ), f"CDF must have 201 points, got {len(final_prediction)}"

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


# Submitting a forecast
def main():
    """Main function to process questions and submit forecasts."""
    posts = list_questions()
    # posts = {"results": [get_question_details(14333)]}

    # Create mapping of post IDs to questions
    post_dict = {}
    for post in posts["results"]:
        if question := post.get("question"):
            post_dict[post["id"]] = [question]
            logging.info(
                f'Post ID: {post["id"]}, Question ID: {post["question"]["id"]}'
            )

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

            if question_details is None:
                logging.info(f"Skipping question {post_id} - forecast already exists")
                continue
            
            question_type = question_details["question"]["type"]

            # Get news context
            formatted_articles = get_formatted_asknews_context(
                question_details["title"]
            )
            log_question_news(
                post_id, formatted_articles, question_details["question"]["title"]
            )

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
                    gpt_result = get_binary_gpt_prediction(
                        question_details, formatted_articles
                    )
                    claude_result = get_binary_claude_prediction(
                        question_details, formatted_articles
                    )

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
                    gpt_result, gpt_comment = get_numeric_gpt_prediction(
                        question_details, formatted_articles
                    )
                    claude_result, claude_comment = get_numeric_claude_prediction(
                        question_details, formatted_articles
                    )

                    if gpt_result and claude_result:
                        gpt_results.append(gpt_result)
                        claude_results.append(claude_result)
                        gpt_texts.append(gpt_comment)
                        claude_texts.append(claude_comment)

                elif question_type == "multiple_choice":
                    # Multiple choice predictions
                    gpt_result, gpt_comment = get_multiple_choice_gpt_prediction(
                        question_details, formatted_articles
                    )
                    claude_result, claude_comment = (
                        get_multiple_choice_claude_prediction(
                            question_details, formatted_articles
                        )
                    )

                    if gpt_result and claude_result:
                        gpt_results.append(gpt_result)
                        claude_results.append(claude_result)
                        gpt_texts.append(gpt_comment)
                        claude_texts.append(claude_comment)

                # Log individual model results
                if gpt_result:
                    log_question_reasoning(
                        post_id,
                        gpt_texts[-1],
                        question_details["question"]["title"],
                        "gpt",
                        run,
                    )
                if claude_result:
                    log_question_reasoning(
                        post_id,
                        claude_texts[-1],
                        question_details["question"]["title"],
                        "claude",
                        run,
                    )

            # Process and submit final prediction if we have results from both models
            if gpt_results and claude_results:
                # Calculate final prediction
                results = {
                    "type": question_type,
                    "gpt_results": gpt_results,
                    "claude_results": claude_results,
                }
                final_prediction = calculate_final_prediction(results, question_details)

                if final_prediction:
                    # Create the API payload
                    forecast_payload = create_forecast_payload(
                        final_prediction[
                            (
                                "prediction"
                                if question_type == "binary"
                                else (
                                    "prediction_probs"
                                    if question_type == "multiple_choice"
                                    else "continuous_cdf"
                                )
                            )
                        ],
                        question_type,
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
                        final_prediction,
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
                            probs_text = "\n".join(
                                [
                                    f"{option}: {prob*100:.2f}%"
                                    for option, prob in probs.items()
                                ]
                            )
                            comment = f"Summary of prediction:\n\n{summary}\n\nFinal predictions:\n{probs_text}"

                        # Post comment using post_id
                        post_question_comment(post_id, comment)
                        logging.info(
                            f"Successfully submitted prediction for question {post_id}"
                        )

            else:
                logging.warning(
                    f"No valid predictions generated for question {post_id}"
                )

        except Exception as e:
            logging.error(f"Error processing question {post_id}: {str(e)}")
            continue

        # Sleep to avoid rate limiting
        time.sleep(2)


if __name__ == "__main__":
    main()
