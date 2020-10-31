import os
import sys
import pathlib
import argparse
from pdb import set_trace
import string
import pickle
import pathlib

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords, words
from nltk import FreqDist

from sklearn.feature_extraction.text import CountVectorizer
from autocorrect import Speller
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from .. import (
  read_data_from_dir,
  train_gibberish_model,
  avg_transition_prob,
  replace_gibberish,
  replace_gibberish_in_spam,
  visualize_tokens,
  clean,
  bar_plot,
  sort_freq_dist,
  detokenize,
  deleet,
  deunicodify
)

pd.options.mode.chained_assignment = None

def pre_process(text):
  text = ''.join(_.lower() for _ in text if _ not in string.punctuation).split()
  text = [_ for _ in text if len(_) > 2 and  _.lower() not in stopwords.words('english') and _.isalpha()]

  #stemmer = PorterStemmer()
  #text = [stemmer.stem(_) for _ in text]

  return ' '.join(text)

def predict(texts):
  return classifier.predict(vectorizer.transform([detokenize(replace_gibberish(deleet(deunicodify(text)), '', gibberish_matrix, gibberish_threshold)) for text in texts]))

def _predict(texts):
  return classifier.predict(vectorizer.transform([detokenize(deleet(deunicodify(text))) for text in texts]))

def __predict(texts):
  return classifier.predict(vectorizer.transform([detokenize(deleet(text)) for text in texts]))

def ___predict(texts):
  return classifier.predict(vectorizer.transform([detokenize(text) for text in texts]))

def ____predict(texts):
  return classifier.predict(vectorizer.transform(texts))

ham_dir = os.path.join('app', 'data', 'enron', 'ham')
spam_dir = os.path.join('app', 'data', 'enron', 'spam')
csv_path = os.path.join('app', 'detect', 'enron.csv')
gibberish_model_path = os.path.join('app', 'gibberish_detector', 'gibberish_model.pki')

results_dir = os.path.join('app', 'results', 'detect')

hammy_words_viz = os.path.join(results_dir, 'hammy_words.png')
spammy_words_viz = os.path.join(results_dir, 'spammy_words.png')
ham_word_size_viz = os.path.join(results_dir, 'ham_word_size.png')
spam_word_size_viz = os.path.join(results_dir, 'spam_word_size.png')

results_path = os.path.join(results_dir, 'results.csv')
false_positives_results_path = os.path.join(results_dir, 'false_positive_results.csv')
false_negatives_results_path = os.path.join(results_dir, 'false_negative_results.csv')

set_trace()

parser = argparse.ArgumentParser()
parser.add_argument('--ham', help='path to the directory that contains ham emails')
parser.add_argument('--spam', help='path to the directory that contains spam emails')

args = parser.parse_args()

if args.ham:
  if pathlib.Path(os.path.abspath(args.ham)).is_dir():
    ham_dir = args.ham

if args.spam:
  if pathlib.Path(os.path.abspath(args.spam)).is_dir():
    spam_dir = args.spam

if not pathlib.Path(gibberish_model_path).is_file():
  try:
    print('\nGibberish model not found at ' + gibberish_model_path + '! Training it now...')
    train_gibberish_model()
  except:
    print('\nAn error occurred while training the gibberish model')
    sys.exit(1)

gibberish_model = pickle.load(open(gibberish_model_path, 'rb'))
gibberish_matrix = gibberish_model['mat']
gibberish_threshold = gibberish_model['thresh']

print('\nGibberish model loaded!')

if not pathlib.Path(csv_path).is_file():
  print('\n' + csv_path + ' not found! Reading from ' + spam_dir + ' and ' + ham_dir)
  data = read_data_from_dir(spam_dir, 'spam')
  data = read_data_from_dir(ham_dir, 'ham', data)
  data['text'] = data['text'].apply(pre_process)
  data.to_csv(csv_path)
  print('\nData read successfully and written to ' + csv_path)

else:
  print('\nReading data from ' + csv_path + '...')
  data = pd.read_csv(csv_path)

spam_text = ' '.join(data[data['label'] == 'spam']['text'])
ham_text = ' '.join(data[data['label'] == 'ham']['text'])

spam_tokens = nltk.word_tokenize(spam_text)
ham_tokens = nltk.word_tokenize(ham_text)

spam_word_lengths_dist = FreqDist([len(_) for _ in spam_tokens])
ham_word_lengths_dist = FreqDist([len(_) for _ in ham_tokens])

#print('\nAverage size of words in spam mails: ' +  str(sum([len(_) for _ in spam_tokens]) / len(spam_tokens)))
#print('Average size of words in ham mails: ' +  str(sum([len(_) for _ in ham_tokens]) / len(ham_tokens)))

bar_plot(
  *list(zip(*spam_word_lengths_dist.most_common(10))),
  'Word size',
  'Number of words',
  'Plot of the sizes of words in spam mails',
  spam_word_size_viz
)

bar_plot(
  *list(zip(*ham_word_lengths_dist.most_common(10))),
  'Word size',
  'Number of words',
  'Plot of the sizes of words in ham mails',
  ham_word_size_viz
)

visualize_tokens(ham_text, save_to=hammy_words_viz)
visualize_tokens(spam_text, save_to=spammy_words_viz)

# Change 1
data = replace_gibberish_in_spam(data, gibberish_matrix, gibberish_threshold)

vectorizer = CountVectorizer()
sparse = vectorizer.fit_transform(data['text'])

feature_names = vectorizer.get_feature_names()

indices = np.arange(len(data))

x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
  sparse,
  data['label'],
  indices,
  test_size=0.20,
  random_state=0
)

classifier = MultinomialNB()
print('\n\nFitting the model...')
classifier.fit(x_train, y_train)

pred = classifier.predict(x_test)

print('\n\nPrediction complete! Results:\n\n')
print('Confusion matrix\n', confusion_matrix(y_test, pred))
print('\n\nAccuracy: ', accuracy_score(y_test, pred))

out_df = pd.DataFrame()

for i in range(len(pred)):
  out_df = out_df.append(data.loc[idx_test[i]])

out_df['predicted_label'] = pred
out_df.to_csv(results_path)

out_false_positives_df = out_df.loc[(out_df['label'] == 'ham') & (out_df['predicted_label'] == 'spam')]
out_false_negatives_df = out_df.loc[(out_df['label'] == 'spam') & (out_df['predicted_label'] == 'ham')]
out_false_positives_df.to_csv(false_positives_results_path)
out_false_negatives_df.to_csv(false_negatives_results_path)

print('\n\nResults written to ' + results_path)

# Defence tests:
examples = [
  #'Win a chance to receive free gifts today! Valid for a limited time only!',
  'W i n a c h a n c e t o r e c e i v e f r e e g i f t s t o d a y ! V a l i d f o r a l i m i t e d t i m e o n l y hou', # tokenization
  'Ԝіո а сhаոсе tο rесеіⅴе frее ցіftѕ tοⅾау! Ꮩаⅼіⅾ fоr а ⅼіⅿіtеⅾ tіⅿе оոⅼу! hou', # invisible obfuscation
  'Ԝіո а сḣаոсе tο rесеіⅴе frее ģіftѕ tοḋау! Ꮩаⅼіḋ fоr а ⅼіⅿіtеḋ tіⅿе оոⅼу! hou', # weak obfuscation
  'Ẃɩռ ắ ĉẖӑῄćḛ ŧợ ӷȩϛḝίὗę ẛřȅḗ ģῑẛṭś ȶʘḓǻӳ! Ѵǎḻἳḑ ẛổȑ ȁ ƚίṃīṯëɗ էίṃē ṑṉḻӯ! hou', # medium obfuscation
  'Ẃῑῇ ἀ ¢ḫἁἧċé ṯȏ ɼεçėȉύȇ ƒȓēϱ ģǐẛեṡ ṫṍƌȃӳ! Ѷǡŀі₫ ḟṓг ἇ ƚίɱὶէёḍ †ỉḿḛ ӧηɭγ! hou', # strong obfuscation
  'W1n 4 ch4nc3 t0 r3c31v3 fr33 g1fts t0d4y! V4l1d f0r a l1m1t3d t1m3 0nly! hou', # leet obfuscation
  'skadnmaskfm sakfmkamf ksafmkasm safasmki ksfmkfsma kafsnksfm Win a chance to receive free gifts today! Valid for a limited time only! hou' # weak statistical
]

print('\nTesting on the following examples:')

set_trace()

for example in examples:
  print(example)

print('\nPredictions without sanitization:')
print(____predict(examples))

print('\nPredictions with detokenization:')
print(___predict(examples))

print('\nPredictions with detokenization and deleetion:')
print(__predict(examples))

print('\nPredictions with detokenization, deleetion, and deunicodification:')
print(_predict(examples))

print('\nPredictions with detokenization, deleetion, deunicodification, and gibberish removal:')
print(predict(examples))

print('\n')
