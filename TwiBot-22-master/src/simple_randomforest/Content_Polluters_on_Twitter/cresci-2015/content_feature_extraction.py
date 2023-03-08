import pandas as pd
import time
import json
import content_feature as cf
import csv
import spacy
nlp = spacy.load('en_core_web_lg')

def get_content_vector(tweets):
    links = cf.count_urls(tweets)
    mentions = cf.count_mentions(tweets)
    content_vector = {
        "links_ratio":links[0],
        "unique_links_ratio":links[1],
        "mention_ratio":mentions[0],
        "unique_mention_ratio":mentions[1],
        "compression_ratio":cf.zip_ratio(tweets),
        "similarity":cf.count_similarity(nlp, tweets), 
    }
    return content_vector

def process_posts(posts):
    post = posts.replace("'","").strip("[").strip("]")
    post = post.split(", ")
    post = post[:200]
    return post
    
def get_all_tweet():
    df = pd.read_csv('./user_post.csv')
    all_tweet_list = []
    for index, row in df.iterrows():
        tweets_list = process_posts(row['post'])
        all_tweet_list.append(tweets_list)
    return all_tweet_list
    
def to_csv(header, dic, filename):
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerows([dic])

if __name__ == '__main__':
    start = time.time()
    
    all_tweet_list = get_all_tweet()
    filepath = './content_feature.csv'
    
    the_header = ["links_ratio", "unique_links_ratio", "mention_ratio", 
                  "unique_mention_ratio", "compression_ratio", "similarity"]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(the_header)
    
    with open("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2015/node.json") as f:
        data = json.load(f)
    new_dict = {}
    for item in data:
        if item['id'][0] == 'u':
            continue
        else:
            tmp = list(item.values())
            new_dict[tmp[0]] = tmp[1]
            
    for tweets in all_tweet_list:
        length = len(tweets)
        for i in range(length):
            tweets[i] = str(new_dict[tweets[i]])
        content_feature = get_content_vector(tweets)
        header = list(content_feature.keys())
        to_csv(header, content_feature, filepath)

    print(time.time() - start)
        