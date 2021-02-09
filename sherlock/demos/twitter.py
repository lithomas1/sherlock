from typing import List

import twint

def get_claims(username: str, limit: int = 10) -> List[str]:
    conf = twint.Config(**__twint_config)
    conf.Limit = limit
    conf.Username = username
    tweets = conf.Store_object_tweets_list = []
    try:
        twint.run.Search(conf)
        return [__sanitize_tweet_of_links(t.tweet) for t in tweets]
    except ValueError:
        return []

def __sanitize_tweet_of_links(tweet: str) -> str:
    link = tweet.rfind("https://")
    return tweet[:link].strip()


__twint_config = {
    "Store_object": True,
    "Output": None,
    "Hide_output": True
}
