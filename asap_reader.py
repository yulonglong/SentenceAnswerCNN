import random
import codecs
import sys
import nltk
import logging
import re
import numpy as np
import pickle as pk

logger = logging.getLogger(__name__)
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')

def is_number(token):
	return bool(num_regex.match(token))

def tokenize(string):
	tokens = nltk.word_tokenize(string)
	for index, token in enumerate(tokens):
		if token == '@' and (index+1) < len(tokens):
			tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
			tokens.pop(index)
	return tokens


def load_vocab(vocab_path):
	logger.info('Loading vocabulary from: ' + vocab_path)
	with open(vocab_path, 'rb') as vocab_file:
		vocab = pk.load(vocab_file)
	return vocab

def create_vocab(file_path, maxlen, vocab_size, tokenize_text, to_lower):
	logger.info('Creating vocabulary from: ' + file_path)
	if maxlen > 0:
		logger.info('  Removing sequences with more than ' + str(maxlen) + ' words')
	total_words, unique_words = 0, 0
	word_freqs = {}
	with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
		input_file.next()
		for line in input_file:
			tokens = line.strip().split('\t')
			question = tokens[0].strip()
			answer = tokens[1].strip()
			content = question + " " + answer

			if to_lower:
				content = content.lower()
			if tokenize_text:
				content = tokenize(content)
			else:
				content = content.split()
			if maxlen > 0 and len(content) > maxlen:
				continue
			for word in content:
				try:
					word_freqs[word] += 1
				except KeyError:
					unique_words += 1
					word_freqs[word] = 1
				total_words += 1
	logger.info('  %i total words, %i unique words' % (total_words, unique_words))
	import operator
	sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
	if vocab_size <= 0:
		# Choose vocab size automatically by removing all singletons
		vocab_size = 0
		for word, freq in sorted_word_freqs:
			if freq > 1:
				vocab_size += 1
	vocab = {'<pad>':0, '<unk>':1, '<num>':2}
	vcb_len = len(vocab)
	index = vcb_len
	for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
		vocab[word] = index
		index += 1
	return vocab

def read_dataset(file_path, maxlen, vocab, tokenize_text, to_lower, score_index=6, char_level=False):
	logger.info('Reading dataset from: ' + file_path)
	if maxlen > 0:
		logger.info('  Removing sequences with more than ' + str(maxlen) + ' words')
	data_x, data_y = [], []
	num_hit, unk_hit, total = 0., 0., 0.
	maxlen_x = -1
	with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
		input_file.next()
		for line in input_file:
			tokens = line.strip().split('\t')
			question = tokens[0].strip()
			answer = tokens[1].strip()
			content = question + " " + answer
			score = int(tokens[2])
		
			if to_lower:
				content = content.lower()
			if char_level:
				content = list(content)
			else:
				if tokenize_text:
					content = tokenize(content)
				else:
					content = content.split()
			if maxlen > 0 and len(content) > maxlen:
				continue
			indices = []
			if char_level:
				pass
			else:
				for word in content:
					if is_number(word):
						indices.append(vocab['<num>'])
						num_hit += 1
					elif word in vocab:
						indices.append(vocab[word])
					else:
						indices.append(vocab['<unk>'])
						unk_hit += 1
					total += 1
			data_x.append(indices)
			data_y.append(score)
			if maxlen_x < len(indices):
				maxlen_x = len(indices)
	logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
	return data_x, data_y, maxlen_x

def get_data(paths, vocab_size, maxlen, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None, score_index=6):
	train_path, dev_path, test_path = paths[0], paths[1], paths[2]
	
	if not vocab_path:
		vocab = create_vocab(train_path, maxlen, vocab_size, tokenize_text, to_lower)
		if len(vocab) < vocab_size:
			logger.warning('The vocabulary includes only %i words (less than %i)' % (len(vocab), vocab_size))
		else:
			assert vocab_size == 0 or len(vocab) == vocab_size
	else:
		vocab = load_vocab(vocab_path)
		if len(vocab) != vocab_size:
			logger.warning('The vocabulary includes %i words which is different from given: %i' % (len(vocab), vocab_size))
	logger.info('  Vocab size: %i' % (len(vocab)))
	
	train_x, train_y, train_maxlen = read_dataset(train_path, maxlen, vocab, tokenize_text, to_lower)
	dev_x, dev_y, dev_maxlen = read_dataset(dev_path, 0, vocab, tokenize_text, to_lower)
	test_x, test_y, test_maxlen = read_dataset(test_path, 0, vocab, tokenize_text, to_lower)
	
	overal_maxlen = max(train_maxlen, dev_maxlen, test_maxlen)
	
	# sort according to length

	if sort_by_len:
		logger.info('Sorting datasets by length')
		
		def len_argsort(seq):
			return sorted(range(len(seq)), key=lambda x: len(seq[x]))
		
		sorted_index = len_argsort(test_set_x)
		test_set_x = [test_set_x[i] for i in sorted_index]
		test_set_y = [test_set_y[i] for i in sorted_index]

		sorted_index = len_argsort(valid_set_x)
		valid_set_x = [valid_set_x[i] for i in sorted_index]
		valid_set_y = [valid_set_y[i] for i in sorted_index]

		sorted_index = len_argsort(train_set_x)
		train_set_x = [train_set_x[i] for i in sorted_index]
		train_set_y = [train_set_y[i] for i in sorted_index]
	
	return ((train_x,train_y), (dev_x,dev_y), (test_x,test_y), vocab, len(vocab), overal_maxlen)