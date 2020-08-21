import praw
from psaw import PushshiftAPI
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime as dt
import pandas as pd

reddit = praw.Reddit(client_id='XXXX',
                     client_secret='XXXX',
                     user_agent='XXXX')

api = PushshiftAPI(reddit)

start_epoch = int(dt.datetime(2010, 1, 1).timestamp())
end_epoch = int(dt.datetime(2010, 1, 31).timestamp())

submission_results = list(api.search_submissions(after=start_epoch,
                                                 before=end_epoch,
                                                 subreddit='politics',
                                                 filter=['id', 'title', 'num_comments',
                                                         'upvote_ratio'],
                                                 limit=100))

topics_dict = {"title": [],
                "score": [],
                "id": [], "url": [],
                "comms_num": [],
                "up_ratio": [],
                "created": [],
                "body": []}

for submission in submission_results:
    topics_dict["title"].append(submission.title)
    topics_dict["score"].append(submission.score)
    topics_dict["id"].append(submission.id)
    topics_dict["url"].append(submission.url)
    topics_dict["comms_num"].append(submission.num_comments)
    topics_dict["up_ratio"].append(submission.upvote_ratio)
    topics_dict["created"].append(submission.created)
    topics_dict["body"].append(submission.selftext)

topics_data = pd.DataFrame(topics_dict)


def get_date(created):
    return dt.datetime.fromtimestamp(created)


_timestamp = topics_data["created"].apply(get_date)

topics_data = topics_data.assign(timestamp=_timestamp)

df = topics_data

analyzer = SentimentIntensityAnalyzer()

sentiment = df['title'].apply(lambda x: analyzer.polarity_scores(x))
df = pd.concat([df, sentiment.apply(pd.Series)], 1)

df.to_csv('FILENAME.csv', encoding='utf-8', index=False)
