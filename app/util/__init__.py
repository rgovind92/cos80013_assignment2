import os
import sys
import pickle
import urllib.request
import json
import string
from pdb import set_trace

import pandas as pd
import matplotlib.pyplot as plt
import wordninja
import nltk
from nltk import FreqDist
from wordcloud import WordCloud

from ..gibberish_detector import avg_transition_prob

ip_map_path = 'app/util/ip_map.pickle'
geolocation_service = "https://geolocation-db.com/json/{ip}&position=true"

try:
  ip_map = pickle.load(open(ip_map_path, 'rb'))
except (OSError, IOError) as e:
  ip_map = {}

def read_data_from_dir(path,
                       label = None,
                       df = pd.DataFrame(),
                       exclude=['cmds']):
  for entry in os.scandir(path):
    if entry.is_dir():
      df = read_data_from_dir(entry.path, label, df)
    
    elif entry.name not in exclude:
      with open(entry.path, encoding='latin-1') as f:
        if label:
          df = df.append([{
            'name': entry.name,
            'text': f.read(),
            'label': label
          }])
        
        else:
          df = df.append([{ 'name': entry.name, 'text': f.read() }])
  
  return df

def get_location(ip, cache=True):
  if cache and ip in ip_map:
    return ip_map[ip]

  else:
    try:
      with urllib.request.urlopen(geolocation_service.format(ip=ip)) as url:
        location = json.loads(url.read().decode())

        if cache:
          ip_map[ip] = location
    except:
      print('Geolocation service to ' + geolocation_service.format(ip=ip) + ' failed!\n')
      sys.exit(1)

    return location

def get_locations(ips, cache=True):
  out = []
  cache_busted = False

  for ip in ips:
    if cache and ip in ip_map:
      out.append(ip_map[ip])
    else:
      cache_busted = True # There's probably a better way to bust the cache
      location = get_location(ip, cache=cache)
      out.append(location)
      if cache:
        ip_map[ip] = location

  if cache and cache_busted:
    pickle.dump(ip_map, open(ip_map_path, 'wb'))

  return out

def is_public_ip(ip):
  try:
    components = [int(_) for _ in ip.strip().split('.')]
  except ValueError:
    return False

  if components[0] == 10:
    return False
  if components[0] == 172 and components[1] in range(16, 32):
    return False
  if components[0] == 192 and components[1] == 168:
    return False
  if components[0] == 127:
    return False

  return True

def sort_freq_dist(freq_dist):
  keys = sorted(freq_dist.keys())
  values = []

  for k in keys:
    values.append(freq_dist[k])

  return (keys, values)

def bar_plot(x, y, xlabel='Key', ylabel='Value', title='Plot', save_to=None, show=False):
  plt.figure()
  plt.bar(x, y)

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)

  if (save_to):
    plt.savefig(save_to)

  if show:
    plt.show()

def get_hops(data):
  hops = []

  if data and 'Received' in data:
    for received in data['Received']:
      if 'from' in received and 'by' in received:
        source = received.split('from')[1].split('by')[0] # Could crash if by is before from
        if '[' in source and ']' in source:
          ip = source.split('[')[1].split(']')[0].strip() # could crash if ] is before [
          if (is_public_ip(ip)):
            hops.append(ip)

  return hops

def extract_headers(data, headers):
  out = {}

  for header in headers:
    out[header] = []

  for header in data.items():
    if header[0] in headers:
      out[header[0]].append(header[1])

  return out

def detokenize(text):
  return ' '.join(
    wordninja.split(
      ''.join(
        ''.join(
          _.lower() for _ in text if _ not in string.punctuation
        ).split()
      )
    )
  )

def get_mistakes(y_test, pred):
  return [i for i in range(len(pred)) if y_test.values[i] != pred[i]]

def get_sentence_from_sparse_array(feature_names, arr):
  return ' '.join([feature_names[i] for i in range(len(arr)) if arr[i] == 1])

def word_uniq_to_one_corpus(c1, c2):
  fdist_c1 = FreqDist(c1)
  common_c1 = fdist_c1.most_common(10)
  words_in_c1 = len(c1)
  fdist_c2 = FreqDist(c2)
  common_c2 = fdist_c2.most_common(10)
  words_in_c2 = len(c2)

  out = {}

  for w1 in common_c1:
    freq_w1_in_c2 = fdist_c2[w1[0]] if fdist_c2[w1[0]] != 0 else 0.00001

    relative_frequency = (w1[1] / words_in_c1) / (freq_w1_in_c2 / words_in_c2)
    if 'c1' not in out or relative_frequency > out['c1'][1]:
      out['c1'] = (w1[0], relative_frequency)

  for w2 in common_c2:
    freq_w2_in_c1 = fdist_c1[w2[0]] if fdist_c1[w2[0]] != 0 else 0.00001

    relative_frequency = (w2[1] / words_in_c2) / (freq_w2_in_c1 / words_in_c1)

    if 'c2' not in out or relative_frequency > out['c2'][1]:
      out['c2'] = (w2[0], relative_frequency)

  return (out['c1'][0], out['c2'][0])

def unusual_words(text):
  text_vocab = set(w.lower() for w in text if w.isalpha())
  english_vocab = set(w.lower() for w in nltk.corpus.words.words())
  unusual = text_vocab - english_vocab
  return sorted(unusual)

def visualize_tokens(text, save_to=False, show=False):
  # Visualisation can get slow when the text is too long
  if len(text.split()) < 400000:
    wc = WordCloud(width=512, height=512).generate(text)
    plt.figure(figsize=(10, 8), facecolor='k')
    plt.imshow(wc)
    plt.axis('off')
    plt.tight_layout(pad=0)

    if save_to:
      plt.savefig(save_to)

    plt.show()

def replace_gibberish(data, replacement, gibberish_matrix, threshold):
  return ' '.join([replacement if avg_transition_prob(_, gibberish_matrix) <= threshold else _ for _ in data.split()])

def replace_gibberish_in_spam(df, gibberish_matrix, threshold):
  df_ham = df[df['label'] == 'ham']
  df_spam = df[df['label'] == 'spam']

  ham = ' '.join(df_ham['text']).split()
  spam = ' '.join(df_spam['text']).split()

  uniq = word_uniq_to_one_corpus(ham, spam)

  df_spam['text'] = df_spam['text'].apply(replace_gibberish, replacement=uniq[1], gibberish_matrix=gibberish_matrix, threshold=threshold)
  return pd.concat([df_spam, df_ham])
