"""
Author: Arie Benhamou
"""

import pandas as pd
import re
from collections import Counter
from operator import itemgetter
import os
import argparse

def is_url_valid(url):
# Checks if the url in the database has a valid format
    if url[:4] != "http":
        return False
    return True
    

def domain_without_www(domain):
# Remove the "www." from urls
    if domain[:4] == "www.":
        return domain[4:]
    return domain


def filter_rows(df):
# Removes the invalid url from the database
    df['is_valid'] = df.url.apply(is_url_valid)
    df = df[df['is_valid'] == True]
    df = df.drop(['is_valid'], axis=1)

# Removes "www." from urls
    df['domain'] = df.domain.apply(domain_without_www)

# Removes NA values
    df = df.dropna()

# Sorts by date
    df = df.sort_values('visit_time_ms', ascending=False)

# Return the cleaned database
    return df.reset_index().drop('index', axis=1)


def remove_this_domain(df, domains):
# Filter domains listed in the FILTERED_DOMAINS variable
    for domain in domains:
        df = df.loc[df['domain'] != domain]
    return df.reset_index().drop('index', axis=1)
    

def clean_date(df):   
    df.day = [re.sub('[-:]', '/', df.day[i]) for i in range(len(df))]
    
    for i in range(len(df)):
        try:
            df.day[i] = pd.to_datetime(df.day[i] + '-'+ df.time[i])
        except ValueError:
            df.day[i] = None
    return filter_rows(df.drop('time', axis=1))


def find_indexes(df, url):
# Gets the indexes of an url in the database
    return df.loc[df['url'] ==  url].index.tolist()


def replace_same_url(df, url, indexes):
# If in the NUM_OF_NEXT visited urls, we find the same url, we remove it and add the next visited url
    index = 0
    while index < len(indexes):
        if df.url[indexes[index]] == url:
            indexes.pop(index)
            last_i = indexes[-1] + 1
            while df.url[last_i] == url and last_i != df.index.max():
                last_i += 1
            indexes += [last_i]
        else:
            index += 1
    return indexes


def weighted_urls(df, url, previous_url, NUM_OF_NEXT=5, NUM_OF_PREVIOUS=2):
# This function gives weights to each URL, depending on how far it's been visited after our current url

# Gets indexes of the NUM_OF_NEXT urls visited after our current url in the database
    indexes_after = [range(element - NUM_OF_NEXT, element) for element in find_indexes(df, url) if element - NUM_OF_NEXT > 0]

# Apply the replace_same_url function
    for index in range(len(indexes_after)):
        indexes_after[index] = replace_same_url(df, url, indexes_after[index])

# Gets indexes of the NUM_OF_BEFORE urls visited before our current url in the database
    indexes_before = [range(element + 1, element + 1 + NUM_OF_PREVIOUS) for element in find_indexes(df, url)]

# If it's the first time we're on current_website, indexes_after will be empty. We take care of this case
    if indexes_after:
        if indexes_after[0] == []:
            indexes_after.pop(0)

# Gets the corresponding url according to the found indexes
    urls_before = [df.url[i].values  for i in indexes_before]
    urls_after = [df.url[i].values  for i in indexes_after]

# Cre
    urls_weighted = []
    for j in range(len(urls_after)):
# Defines the weights given to each url visited after current_website : [1, 0.5, 0.25, 0.125...]
        weights = [float(1)/2**(NUM_OF_NEXT - i -1) for i in range(NUM_OF_NEXT)]
# If the url visited before current_website matches with the current url we're working with, multiply the weights by 2
# Same if the previous website also matches
        i = 0
        while i < NUM_OF_PREVIOUS and previous_url[i] == urls_before[j][i]:
            weights = [weight*2 for weight in weights]
            i += 1
        urls_weighted.append([(urls_after[j][k], weights[k]) for k in range(len(weights))])           
    return urls_weighted


def get_weigths(df, url, previous_url):
# For each unique url, sum the weights, sorts the result and return the sorted dictionnary    
    weigth_dic = {}
    for url_list in weighted_urls(df, url, previous_url):
        for url in url_list:
            if url[0] in weigth_dic:
                weigth_dic[url[0]] += url[1]
            else:
                weigth_dic[url[0]] = url[1]

    return sorted(weigth_dic.items(), key=itemgetter(1), reverse=True)


def to_int(number):
# Allows to treat dates as int
    if not isinstance(number, float) and "," in number:
        number = number.replace(",", ".")
    return float(number)


def delta(df):
# Creates a new column, in which we can find the time spent on the url before opening a new url
    df.visit_time_ms = df.visit_time_ms.apply(to_int)
    df = df.sort_values('visit_time_ms', ascending = False)
    df['delta'] = [df.visit_time_ms[index] - df.visit_time_ms[index + 1] for index in range(df.shape[0] - 1)] + [0]
    return df


def treat_data(df_path, filtered_domains):
    df = pd.read_csv(df_path, delimiter=";", header=None)
    df.columns = (['url', 'domain', 'root domain', 'visit_time_ms', 'visit_time_str', 'day of the week', 'transition_type', 'page title'])
    df = df[['url', 'domain', 'visit_time_ms', 'visit_time_str', 'transition_type']]
    df = filter_rows(df)
    df = remove_this_domain(df, filtered_domains)
    return df


def get_other_databases(folder_path, filtered_domains):
    df_list = []
    for df in os.listdir(folder_path):
        if os.path.basename(df)[0] != ".":
            df_list.append(treat_data(folder_path + os.path.basename(df), filtered_domains))
    return df_list


def sum_score(df_list, arg2, arg3):
    score_dic = {}
    for dataframe in df_list:
        for i in get_weigths(dataframe, arg2, arg3):
            if i[0] in score_dic:
                score_dic[i[0]] += i[1]
            else:
                score_dic[i[0]] = i[1]

    return sorted(score_dic.items(), key=itemgetter(1), reverse=True)

def print_most_common():
    list_set = [set(df.url.values)]
    for dataframe in df_list:
        list_set.append(list(set(dataframe.url.values)))
    a = [item for sublist in list_set for item in sublist]
    b = Counter(a)
    for i in b.most_common()[:5]:
        print i

def get_results_yourself(df_yourself, current_url, previous_urls):
    best_recommandations = [i[0] for i in get_weigths(df_yourself, current_url, previous_urls)]
    return best_recommandations[:5]


def get_results_others(df_list, current_url, previous_urls):
    best_recommandations = [i[0] for i in sum_score(df_list, current_url, previous_urls)]
    return best_recommandations[:5]

def split_website_videos(results):
    websites = []
    videos = []
    for result in results:
        if re.search(r"^https://www.youtube.com/watch", result):
            videos.append(result)
        else:
            websites.append(result)
    return websites, videos

def generate_results(df_yourself, df_list, current_url, previous_urls, others="Results/others.txt", yourself="Results/yourself.txt", youtube="Results/youtube.txt"):
    
    websites, videos = split_website_videos(get_results_others(df_list, current_url, previous_urls))
    
    file = open(others, "w")
    results_others = []
    for url in websites[:3]:
        file.write(url)
        file.write(os.linesep)
    file.close()

    file = open(yourself, "w")
    for url in [get_results_yourself(df_yourself, current_url, previous_urls)[0]]:
        file.write(url)
        file.write(os.linesep)
    file.close()

    results_youtube = []
    file = open(youtube, "w")
    for url in videos:
        file.write(url)
        file.write(os.linesep)
    file.close()


def main(current_url, previous_urls):

    if not os.path.exists("Results/"):
        os.makedirs("Results/")

    FILTERED_DOMAINS = (['whatsapp.com','web.whatsapp.com', 'twitter.com', 'linkedin.com',
                         'google.co.il','fr-fr.messenger.com','youtube.com','facebook.com', 'localhost',
                         'plus.google.com','google.fr', 'mail.google.com', 'google.com','messenger.com',
                         'listenonrepeat.com', 'drive.google.com', 'docs.google.com', 'calendar.google.com',
                         'chrome.google.com', 'gmail.com', 'lefigaro.fr'])

    DF_PATH = 'Databases/my_database.csv'
    FOLDER_PATH = 'Databases/other_databases/'

    df = treat_data(DF_PATH, FILTERED_DOMAINS)
    df_list = get_other_databases(FOLDER_PATH, FILTERED_DOMAINS)

    others = "Results/others.txt"
    yourself = "Results/yourself.txt"
    youtube = "Results/youtube.txt"

    generate_results(df, df_list, current_url, previous_urls)

if __name__=='__main__':

    current_url, previous_urls = 'http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html', [ "", ""]

    # parser = argparse.ArgumentParser()
    # parser.add_argument("current_url",type=str,help="Current url we're visiting")
    # parser.add_argument("previous_urls",type=list,help="Previous urls we've just visited")
    # args = parser.parse_args()

    # arg2 = args.current_url
    # arg3 = args.previous_urls

    main(current_url, previous_urls)


