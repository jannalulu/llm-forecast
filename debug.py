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
OPENAI_API_KEY = secrets['OPENAI_API_KEY']
ASKNEWS_CLIENT_ID = secrets['ASKNEWS_CLIENT_ID']
ASKNEWS_SECRET = secrets['ASKNEWS_SECRET']

from asknews_sdk import AskNewsSDK

def post_question_comment(question_id, comment_text):
    """
    Post a comment on the question page as the bot user.
    """
    response = sdk.news.search_news(
        query="kamala harris",
        n_articles=10,
        return_type="both",
        similarity_score_threshold=0.01,
        method="kw",
    )
    print(response.as_string)
    with open("search.json", "w") as f:
        f.write(response.model_dump_json())