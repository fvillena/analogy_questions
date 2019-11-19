import numpy as np
import pandas as pd
import multiprocessing
import json
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser=argparse.ArgumentParser()

parser.add_argument('--lang', help='language')
parser.add_argument('--limit', help='limit the number of rows to read')
parser.add_argument('--threads', help='number of threads to use')

args=parser.parse_args()

LANG = str(args.lang)
LIMIT = args.limit
if LIMIT != None:
    LIMIT = int(LIMIT)
THREADS = int(args.threads)

def cui2words(cui):
    words = []
    pt = umls[(umls.CUI == cui)
             & (umls.LAT == LANG)
             & (umls.ISPREF == 'Y')
             & (umls.TTY == 'PT')
             & (umls.TS == 'P')
             & (umls.SUPRESS == 'N')
            ]["STR"].to_list()
    pt = [t.lower() for t in pt]
    pt = [t for t in pt if t.isalpha()]
    pt = list(set(pt))
    terms = umls[(umls.CUI == cui)
                 & (umls.LAT == LANG)
                 & (umls.SUPRESS == 'N')
                ]["STR"].to_list()
    terms = [term.lower() for term in terms]
    terms = [term for term in terms if not term in pt]
    terms = [term for term in terms if term.isalpha()]
    terms = list(set(terms))
    words.extend(pt)
    words.extend(terms)
    return words

def word2cui(word):
    result = list(set(umls[(umls['STR'] == word) & (umls.SUPRESS == 'N')].CUI))
    if len(result) == 1:
        return result[0]
    else:
        return None

def list2cui(word_list):
    cui_list = [word2cui_dict[word] for word in word_list]
    if None in cui_list:
        return []
    else:
        return (cui_list)

def list2words(cui_list):
    word_list = [tuple(cui2words_dict[cui]) for cui in cui_list]
    return word_list

logger.info('reading umls table')

umls = pd.read_csv('MRCONSO.RRF', sep='|', nrows=LIMIT)

logger.info('reading analogy test set')

questions = {}
vocab = []
with open('medical_analogy_test_set_en.txt', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip()
        if line.startswith(':'):
            cat = line[2:]
        else:
            words = line.split(' ')
            vocab.extend(words)
            if cat in questions:
                questions[cat].append(words)
            else:
                questions[cat] = [words]
vocab = list(set(vocab))

logger.info('translating vocab to cui')

pool = multiprocessing.Pool(THREADS)
translated_vocab = pool.map(word2cui,vocab)

word2cui_dict = {word:cui for word,cui in zip(vocab,translated_vocab)}

logger.info('translating cui to words')

pool = multiprocessing.Pool(THREADS)
translated_vocab_words = pool.map(cui2words,translated_vocab)

cui2words_dict = {cui:words for cui,words in zip(translated_vocab,translated_vocab_words)}

logger.info('extracting relations')

relations = {}
for key,val in questions.items():
    if not key in relations:
        relations[key] = []
    for question in val:
        relations[key].append(question[:2])
        relations[key].append(question[2:])

logger.info('translating relations')

translated_relations = {}
for key,val in relations.items():
    pool = multiprocessing.Pool(THREADS)
    translated_relations[key] = {}
    translated_relations[key]['cui'] = pool.map(list2cui,val)
    translated_relations[key]['cui'] = [translated_relation for translated_relation in translated_relations[key]['cui'] if len(translated_relation) > 0]
    translated_relations[key]['cui'] = list(set(tuple(relation) for relation in translated_relations[key]['cui']))
    
    translated_relations[key]['words'] = pool.map(list2words,translated_relations[key]['cui'])
    translated_relations[key]['words'] = [relation for relation in translated_relations[key]['words'] if ( (len(relation[0]) > 0) & (len(relation[1]) > 0) )]

logger.info('writing relations')

with open('relations_{}.json'.format(LANG), 'w') as outfile:
    json.dump(translated_relations, outfile, ensure_ascii=False, indent=2)