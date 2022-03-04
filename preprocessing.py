import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from textblob import TextBlob
# !pip install contractions
import contractions
from copy import deepcopy
import torch


class TextDataPreprocessor():
	"""
			This class is used to preprocess a corpus (a list of 
		strings (texts)) for nlp tasks. Preprocessing contains
		cleansing the texts and building data structures needed 
		in further nlp tasks.
			The methods provided to cleanse the texts include:
			(1) Chop off all characters that are absent in a 
			defined library.
			(2) Lower the case of all characters.
			(3) Apply a customized mapping to each text string via 
			a provided map.
			(4) Expand all contractions.
			(5) Fix the misspellings.
			(6) Remove selected punctuations and remain the others.
			The data structures that can be constructed by the 
		provided methods include:
			(1) Perform word-level tokenization on a text corpus 
			to get a corresponding corpus in the form of tokens 
			(A list of lists of tokens).
			(2) Build a sorted vocabulary (sorted_vocab) containing 
			(word, num_word) pairs for all words (list of tuples).
			(3) Build a word2idx map mapping each word to an integer.
			(4) Build a list containing integer sequences with each 
			sequence stored as a sublist. Each integer is mapped from 
			a token via the word2idx map.
			(5) The sequences in (4) can be chosen to have at least 
			100 elements by padding with 0 from either the front or
			the end.
			(6) Build a embedding matrix.
	"""

	def __init__(self, max_num_words = None,
				 min_seq_length = None,
				 front_padded = False,
				 chop = False,
				 lower = True,
				 contracted = False,
				 fix_misspelling = False,
				 to_remove = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
				 customized_map = None):
			
		self.max_num_words = max_num_words
		self.min_seq_length = min_seq_length
		self.front_padded = front_padded
		self.chop = chop
		self.lower = lower
		self.contracted = contracted
		self.fix_misspelling = fix_misspelling
		self.to_remove = to_remove
		self.to_keep = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
		self.customized_map = customized_map
		self.character_lib = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
				 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
				 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
				 '!', '"', '\'', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', 
				 '^', '_', '`', '{', '|', '}', '~', '\t', '\n', ' ', '', "’", "‘", "´", "`")
		self.tokenized_corpus = list()
		self.sorted_vocab = list()
		self.word2idx = dict()
		self.sequences = list()
		self.padded_sequences = None
		self.num_words = int()

		for punc in self.to_remove:
			self.to_keep = self.to_keep.replace(punc, '')

	def cleanse_corpus(self, corpus):
		"""
				Cleanse the corpus.
		"""
		corpus = pd.Series(corpus)

		if self.chop:
			corpus = corpus.apply(lambda x: self.chop_off(x))
		if self.customized_map:
			corpus = corpus.apply(lambda x: self.customized_cleansing(x))
		if not self.contracted:
			corpus = corpus.apply(lambda x: self.expand_contractions(x))
		if self.lower:  # Place lower_case() after expand_contractions() because contractions.fit() doesn't preserve the case of some words
			corpus = corpus.apply(lambda x: self.lower_case(x))
		if self.fix_misspelling:
			corpus = corpus.apply(lambda x: self.corret_misspelling(x))
		corpus = corpus.apply(lambda x: self.process_punctuations(x))

		return corpus

	def chop_off(self, text):
		for character in text:
			if character not in self.character_lib:
				text = text.replace(character, ' ')
		return text

	def lower_case(self, text):
		text = text.lower()
		return text

	def expand_contractions(self, text):
		for item in ["’", "‘", "´", "`"]:
			text = text.replace(item, "'")
		try:
			text = contractions.fix(text)
		except IndexError:
			pass
		return text

	def corret_misspelling(self, text):
		text = str(TextBlob(text).correct())
		return text

	def customized_cleansing(self, text):
		for item in self.customized_map:
			text = text.replace(item, self.customized_map[item])
		return text

	def process_punctuations(self, text):
		for punc in self.to_keep:
			text = text.replace(punc, f' {punc} ')

		for punc in self.to_remove:
			text = text.replace(punc, ' ')
		return text

	def fit_on_corpus(self, corpus):
		"""
				Fit the tokenizer on a corpus, which is 
			a list of strings (texts).
		"""
		self.tokenized_corpus = self.tokenize_corpus(corpus)
		self.sorted_vocab = self.build_sorted_vocab(self.tokenized_corpus)
		self.word2idx = self.build_word2idx(self.sorted_vocab)
		self.sequences = self.texts_to_sequences(self.tokenized_corpus, self.word2idx)
		if self.min_seq_length:
			self.padded_sequences = self.pad_sequence(self.sequences)

	def tokenize_corpus(self, corpus): 
		return [[j for j in i.split() if j] for i in corpus]

	def build_sorted_vocab(self, tokenized_corpus):
		vocab = defaultdict(int)
		for text in tokenized_corpus:
			for token in text:
				vocab[token] += 1
		sorted_vocab = list(vocab.items())
		sorted_vocab.sort(key=lambda x: x[1], reverse=True)
		return sorted_vocab

	def build_word2idx(self, sorted_vocab):
		word_list = []
		word_list.extend(item[0] for item in sorted_vocab)
		return dict( zip(word_list, list(range(1, len(word_list) + 1))) )

	def texts_to_sequences(self, tokenized_corpus, word2idx):
		return list(self.texts_to_sequences_generator(tokenized_corpus, word2idx))

	def texts_to_sequences_generator(self, tokenized_corpus, word2idx):
		for word_seq in tokenized_corpus:
			seq = list()
			for token in word_seq:
				idx = word2idx.get(token)
				if idx is not None:
					if self.max_num_words and idx >= self.max_num_words:
						pass
					else:
						seq.append(idx)
				else:
					pass
			yield seq

	def pad_sequence(self, sequences):
		padded_sequences = deepcopy(list(sequences))
		for index in range(len(padded_sequences)):
			if len(padded_sequences[index]) < self.min_seq_length:
				if self.front_padded:
					padded_sequences[index] = [0 for i in range(self.min_seq_length - len(padded_sequences[index]))] + padded_sequences[index]
				else:
					padded_sequences[index].extend([0 for i in range(self.min_seq_length - len(padded_sequences[index]))])
			else:
				padded_sequences[index] = padded_sequences[index][:self.min_seq_length]
		return padded_sequences

	def build_embedding_matrix(self, embedding_dim, word_embeddings):
		"""
				Required inputs:
				1. embedding_dim is the embedding dimension
				2. word_embeddings is a dict containing the embeddings
				of words.
		"""
		if self.max_num_words:
			self.num_words = min(self.max_num_words, len(self.sorted_vocab))
		else:
			self.num_words = len(self.sorted_vocab)
		embedding_matrix = np.zeros((self.num_words, embedding_dim))

		for word, idx in self.word2idx.items():
			if idx < self.num_words:
				word_vector = word_embeddings.get(word)
				if word_vector is not None:
					embedding_matrix[idx] = word_vector
				else:
					pass
			else:
				pass
		
		return embedding_matrix

	def get_sequences_of_test_texts(self, texts):
		"""
				Turn test texts (list of strings) into sequences (list
			of sequence).
		"""
		tokenized_texts = self.tokenize_corpus(texts)
		seqs = self.texts_to_sequences(tokenized_texts, self.word2idx)
		if self.min_seq_length:
			return self.pad_sequence(seqs)
		return seqs


def corpus_to_vocab(corpus):
	tokenized_corpus = [[j for j in i.split() if j] for i in corpus]
	vocab = defaultdict(int)
	for text in tokenized_corpus:
		for token in text:
			vocab[token] += 1
	return vocab


def embedding_coverage(vocab, word2embedding):
	known_words = defaultdict(int)
	unknown_words = defaultdict(int)

	words_with_embedding = word2embedding.keys()

	for word in vocab.keys():
		if word in words_with_embedding:
			known_words[word] = vocab[word]
		else:
			unknown_words[word] = vocab[word]
			
	num_known_words = sum(known_words.values())
	num_unknown_words = sum(unknown_words.values())

	print('Embeddings founded for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
	print('Embeddings founded for {:.2%} of all texts'.format(num_known_words / (num_known_words + num_unknown_words)))
	
	sorted_unknown_words = list(unknown_words.items())
	sorted_unknown_words.sort(key=lambda x: x[1], reverse=True)
	
	return sorted_unknown_words



class SDDataset(torch.utils.data.Dataset):
	def __init__(self, X, y):
		super().__init__()
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return (torch.tensor(self.X[idx], dtype = torch.int64), torch.tensor(self.y[idx], dtype = torch.float32))










