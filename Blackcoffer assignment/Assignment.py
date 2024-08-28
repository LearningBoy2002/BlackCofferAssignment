import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_excel('Input.xlsx')

url_id_map = dict(zip(df['URL_ID'].astype(str) + '.txt', df['URL_ID']))

if not os.path.exists('TitleText'):
    os.makedirs('TitleText')


for index, row in df.iterrows():
    url = row['URL']
    url_id = row['URL_ID']

    header = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}
    try:
        response = requests.get(url, headers=header)
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('h1').get_text()
        article = " ".join([p.get_text() for p in soup.find_all('p')])

        file_name = f'TitleText/{url_id}.txt'
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(title + '\n' + article)
    except Exception as e:
        print(f"Error processing URL_ID {url_id}: {str(e)}")


stop_words = set(stopwords.words('english'))
for file in os.listdir('StopWords'):
    with open(os.path.join('StopWords', file), 'r', encoding='ISO-8859-1') as f:
        stop_words.update(set(f.read().splitlines()))


positive_words = set()
negative_words = set()
with open('MasterDictionary/positive-words.txt', 'r', encoding='ISO-8859-1') as f:
    positive_words.update(f.read().splitlines())
with open('MasterDictionary/negative-words.txt', 'r', encoding='ISO-8859-1') as f:
    negative_words.update(f.read().splitlines())

def analyze_sentiment(words):
    positive_score = sum(1 for word in words if word.lower() in positive_words)
    negative_score = sum(1 for word in words if word.lower() in negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score


def analyze_readability(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    clean_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    
    avg_sentence_length = len(clean_words) / len(sentences)
    complex_words = [word for word in clean_words if sum(1 for char in word if char.lower() in 'aeiou') > 2]
    percent_complex_words = len(complex_words) / len(clean_words)
    fog_index = 0.4 * (avg_sentence_length + percent_complex_words)
    
    syllable_count = sum(sum(1 for char in word if char.lower() in 'aeiou') for word in clean_words)
    avg_syllable_word_count = syllable_count / len(clean_words)
    
    word_count = len(clean_words)
    avg_word_length = sum(len(word) for word in clean_words) / word_count
    
    personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', text, re.IGNORECASE))
    
    return avg_sentence_length, percent_complex_words, fog_index, len(complex_words), avg_syllable_word_count, word_count, avg_word_length, personal_pronouns

results = []
for file in os.listdir('TitleText'):
    if file.endswith('.txt'):  # Ensure we're only processing text files
        with open(os.path.join('TitleText', file), 'r', encoding='utf-8') as f:
            text = f.read()
        words = word_tokenize(text.lower())
        clean_words = [word for word in words if word not in stop_words and word.isalnum()]
        
        pos_score, neg_score, polarity, subjectivity = analyze_sentiment(clean_words)
        avg_sent_len, perc_complex, fog, complex_count, avg_syllable, word_count, avg_word_len, pronoun_count = analyze_readability(text)
        
        # Use the mapping to get the correct URL_ID
        url_id = url_id_map.get(file, None)
        
        if url_id is not None:
            results.append({
                'URL_ID': url_id,
                'POSITIVE SCORE': pos_score,
                'NEGATIVE SCORE': neg_score,
                'POLARITY SCORE': polarity,
                'SUBJECTIVITY SCORE': subjectivity,
                'AVG SENTENCE LENGTH': avg_sent_len,
                'PERCENTAGE OF COMPLEX WORDS': perc_complex,
                'FOG INDEX': fog,
                'AVG NUMBER OF WORDS PER SENTENCE': avg_sent_len,
                'COMPLEX WORD COUNT': complex_count,
                'WORD COUNT': word_count,
                'SYLLABLE PER WORD': avg_syllable,
                'PERSONAL PRONOUNS': pronoun_count,
                'AVG WORD LENGTH': avg_word_len
            })
        else:
            print(f"Warning: No matching URL_ID found for file {file}")


output_df = pd.DataFrame(results)
final_df = pd.merge(df, output_df, on='URL_ID')
final_df.to_csv('Output_Data.csv', index=False)