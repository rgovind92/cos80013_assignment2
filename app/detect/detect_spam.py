import pathlib
import sys
import os
from email.parser import HeaderParser
from pdb import set_trace
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from nltk import FreqDist
import pickle

from .. import (
  read_data_from_dir,
  get_location,
  get_locations,
  is_public_ip,
  sort_freq_dist,
  bar_plot,
  get_hops,
  extract_headers
)

pd.options.mode.chained_assignment = None

results_dir = os.path.join('app', 'results', 'trace')

ham_dir = os.path.join('app', 'data','spam_assassin','ham')
spam_dir = os.path.join('app', 'data','spam_assassin','spam')

ham_file_path = os.path.join('app','trace','ham.csv')
spam_file_path = os.path.join('app','trace','spam.csv')

ham_file = pathlib.Path(ham_file_path)
spam_file = pathlib.Path(spam_file_path)

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

print('\n\n')

# If ham or spam csv does not exist, read it recursively from
# its respective directories, write the result to a csv,
# and use that csv in subsequent runs

if ham_file.is_file():
  print('\n\nReading data from ' + ham_file_path + '...')
  ham = pd.read_csv(ham_file_path)
else:
  try:
    print('\n\n' + ham_file_path + ' not found! Reading from ' + ham_dir)
    ham = read_data_from_dir(ham_dir)
  except:
    print('Ham folder (' + ham_dir + ') could not be accessed!\n')
    sys.exit(1)

  ham.to_csv(ham_file_path)
  print('\n\nData read successfully and written to ' + ham_file_path

if spam_file.is_file():
  print('\n\nReading data from ' + spam_file_path + '...')
  spam = pd.read_csv(spam_file_path)
else:
  try:
    print('\n\n' + spam_file_path + ' not found! Reading from ' + spam_dir)
    spam = read_data_from_dir(spam_dir)
  except:
    print('Spam folder could not be accessed!\n')
    sys.exit(1)  

  print('\n\nData read successfully and written to ' + spam_file_path
  spam.to_csv(spam_file_path)

print('\n\nExtracting headers...')

ham_headers = [extract_headers(HeaderParser().parsestr(_), ['Received', 'From']) for _ in ham['text']]
spam_headers = [extract_headers(HeaderParser().parsestr(_), ['Received', 'From']) for _ in spam['text']]

ham_headers_received_len_freq_dist = FreqDist([len(_['Received']) for _ in ham_headers])
spam_headers_received_len_freq_dist = FreqDist([len(_['Received']) for _ in spam_headers])

ham_hop_locations = [get_locations(get_hops(_)) for _ in ham_headers]
spam_hop_locations = [get_locations(get_hops(_)) for _ in spam_headers]

ham_hop_countries = [[_['country_name'] for _ in __] for __ in ham_hop_locations]
spam_hop_countries = [[_['country_name'] for _ in __] for __ in spam_hop_locations]

ham_hop_countries_uniq = [len(set(_)) for _ in ham_hop_countries]
spam_hop_countries_uniq = [len(set(_)) for _ in spam_hop_countries]

ham_origin_countries = [_[-1] if _ and len(_) > 0 else 'Unknown' for _ in ham_hop_countries]
spam_origin_countries = [_[-1] if _ and len(_) > 0 else 'Unknown' for _ in spam_hop_countries]

ham_origin_addresses = [(_[-1]['city'] or 'Unknown city') + ', ' + (_[-1]['state'] or 'Unknown state') + ', ' + (_[-1]['country_name'] or 'Unknown country') if len(_) > 0 else 'Unknown' for _ in ham_hop_locations]
spam_origin_addresses = [(_[-1]['city'] or 'Unknown city') + ', ' + (_[-1]['state'] or 'Unknown state') + ', ' + (_[-1]['country_name'] or 'Unknown country') if len(_) > 0 else 'Unknown' for _ in spam_hop_locations]

ham_origin_countries_freq_dist = FreqDist(ham_origin_countries)
spam_origin_countries_freq_dist = FreqDist(spam_origin_countries)

ham_hop_countries_uniq_freq_dist = FreqDist(ham_hop_countries_uniq)
spam_hop_countries_uniq_freq_dist = FreqDist(spam_hop_countries_uniq)

bar_plot(
  *sort_freq_dist(ham_headers_received_len_freq_dist),
  'Number of hops',
  'Number of emails',
  'Plot of number of Received headers in ham emails',
  os.path.join(results_dir, 'ham_received.png')
)
bar_plot(
  *sort_freq_dist(spam_headers_received_len_freq_dist),
  'Number of hops',
  'Number of emails',
  'Plot of number of Received headers in spam emails',
  os.path.join(results_dir, 'spam_received.png')
)

bar_plot(
  *list(zip(*ham_hop_countries_uniq_freq_dist.most_common(10))),
  'Number of unique countries hopped',
  'Number of emails',
  'Plot of the number of unique countries through which ham emails hopped',
  os.path.join(results_dir, 'ham_hop_unique_countries.png')
)

bar_plot(
  *list(zip(*spam_hop_countries_uniq_freq_dist.most_common(10))),
  'Number of unique countries hopped',
  'Number of emails',
  'Plot of the number of unique countries ehrough which spam emails hopped',
  os.path.join(results_dir, 'spam_hop_unique_countries.png')
)

ham = ham[['name']]
spam = spam[['name']]

ham['origin_address'] = ham_origin_addresses
spam['origin_address'] = spam_origin_addresses

ham.to_csv(os.path.join(results_dir, 'ham_origins.csv'))
spam.to_csv(os.path.join(results_dir, 'spam_origins.csv'))

print('\n\nSuccess! Results have been saved to ' + results_dir)
