import gensim
import numpy as np
import json
import itertools
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser=argparse.ArgumentParser()

parser.add_argument('--model', help='model location')
parser.add_argument('--limit', help='limit the model vocabulary')
parser.add_argument('--dataset', help='dataset location')
parser.add_argument('--k', help='k')
parser.add_argument('--binary', help='binary model')
parser.add_argument('--normalize', help='strip accents')

args=parser.parse_args()

MODEL_LOCATION = args.model
DATASET_LOCATION = args.dataset
K = int(args.k)
LIMIT = args.limit
if LIMIT != None:
    LIMIT = int(LIMIT)
if args.binary == "True":
    BINARY = True
if args.binary == "False":
    BINARY = False
if args.normalize == "True":
    NORMALIZE = True
if args.normalize == "False":
    NORMALIZE = False 

logger.info("arguments are: {}".format(args))

RESULT_LOCATION = "{}-{}-{}-{}-{}_result.json".format(MODEL_LOCATION,DATASET_LOCATION,LIMIT,K,NORMALIZE)

import re
def mreplace(s, chararray, newchararray):
    for a, b in zip(chararray, newchararray):
        s = s.replace(a, b)
    return s
def normalizer(words):
    words = [mreplace(re.sub(r'[^a-zA-Záéíóúñ]', '', word.lower()),'áéíóú','aeiou') for word in words]
    return words
any_in = lambda a, b: any(i in b for i in a)
all_in = lambda a, b: all(i in b for i in a)

def solveAnalogy(model, k, a, b, c):
    """ Receives a word2vec gensim model, a k for the most similar words searching and 3 lists of words.
    The queried words assume the construction of the question 'a' is a 'b' as 'c' is a 'k most similar words'.
    Example:
    a = ["hombre","varón", "macho", "masculino"]
    b = ["mujer","hembra","fémina","dama"]
    c = ["rey"]
    The function returns a list of tuples in the form of (word, distance)
    """

    a_vector = np.median([model[word] for word in a], axis=0)
    b_vector = np.median([model[word] for word in b], axis=0)
    c_vector = np.median([model[word] for word in c], axis=0)

    v = b_vector - a_vector + c_vector
    most_similar = model.most_similar(np.array([v]), topn = k+len(c))
    result = [x for x in most_similar if x[0] not in c]
    return result[:k]

def evaluator(model, k, a, b, c, correct):
    responses = [response[0] for response in solveAnalogy(model, k, a, b, c)]
    return any_in(correct, responses)

class AnalogyEvaluator:
    def __init__(self, k=5):
        self.k = k
    def set_model(self, model):
        logger.info('setting model')
        self.model = model
        self.vocabulary = list(model.vocab.keys())
        logger.info('model has {} words'.format(len(self.vocabulary)))
    def set_dataset(self, dataset_file):
        self.questions = {}
        logger.info('setting dataset')
        with open(dataset_file) as json_file:
            data = json.load(json_file)
        for key,val in data.items():
            relations_in_vocab = []
            for relation in val['words']:
                relation_vocab = list(itertools.chain.from_iterable(relation))
                if all_in(relation_vocab,self.vocabulary):
                    relations_in_vocab.append(relation)
                elif NORMALIZE & all_in(normalizer(relation_vocab),self.vocabulary):
                    relations_in_vocab.append([normalizer(relation[0]),normalizer(relation[1])])
            permutations = itertools.permutations(relations_in_vocab,2)
            self.questions[key] = list(permutations)
            logger.info("{} has {} relations, where {} are in vocabulary. {} has {} questions".format(
                key,len(val['words']),len(relations_in_vocab),key,len(self.questions[key])))
    def evaluate(self):
        logger.info('evaluating model')
        self.correct = {}
        for group, questions in self.questions.items():
            logger.info('evaluating {}'.format(group))
            correct = []
            for question in questions:
                if evaluator(self.model, self.k, question[0][0], question[0][1], question[1][0], question[1][1]):
                    correct.append(question)
            self.correct[group] = correct
        self.performance = {}
        for group,questions in self.questions.items():
            if len(questions) == 0:
                accuracy = 0
            else:
                accuracy = len(self.correct[group])/len(questions)
            self.performance[group] = accuracy

model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_LOCATION, binary=BINARY, limit = LIMIT)
ae = AnalogyEvaluator()
ae.set_model(model)
ae.set_dataset(DATASET_LOCATION)
ae.evaluate()

result = {
    "accuracy" : ae.performance,
    "n_questions" : {key:len(val) for key,val in ae.questions.items()},
    "n_correct_questions" : {key:len(val) for key,val in ae.correct.items()},
    "config" : vars(args),
    "correct_questions" : ae.correct,
    "questions" : ae.questions
}

with open(RESULT_LOCATION, 'w') as outfile:
    json.dump(result, outfile, ensure_ascii=False, indent=2)