import streamlit as st

st.set_page_config(page_title="Home Depot Search Relevance",page_icon="👾",layout="wide")	# Set the page title and icon

st.title("Home Depot Search Relevance")																																			# Set the title of the page
st.text("""Please enter your search query in the text box below to fetch the top 10 relevant products 
and its relevance score""")	# Set the text of the page
import pandas as pd
import numpy as np
import regex as re
# !pip install nltk
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import PorterStemmer
# from wordcloud import STOPWORDS
# from prettytable import PrettyTable
import warnings

warnings.filterwarnings('ignore')
import math
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from numpy.linalg import norm
import pickle
from tqdm.notebook import tqdm
from scipy.stats import uniform, randint, loguniform
# !pip install rank_bm25
from tqdm import tqdm

tqdm.pandas()
from nltk.metrics.distance import edit_distance
from nltk.metrics.distance import jaccard_distance
# from gensim.test.utils import common_texts, get_tmpfile
# from gensim.models import Word2Vec
# from gensim.models.callbacks import CallbackAny2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from flask import Flask, jsonify, request
import json
# from rank_bm25 import BM25Okapi

# import flask

# app = Flask(__name__)
print('Imports Done!')

# """**BM25**"""

database = pd.read_pickle('./model/database.pkl')

product_text = database['text'].values

infile = open('./model/BM25_model.pkl', 'rb')
bm25_model = pickle.load(infile)


# corpus = database['text'].values
# tokenized_corpus = [doc.split(" ") for doc in corpus]
#
# bm25_model = BM25Okapi(tokenized_corpus)


def words(text): return re.findall(r'\w+', text.lower())


WORDS = Counter(words(open('./model/corpus.txt').read()))


def P(word, N=sum(WORDS.values())):
	# "Probability of `word`."
	return WORDS[word] / N


def correction(word):
	# "Most probable spelling correction for word."
	return max(candidates(word), key=P)


def candidates(word):
	# "Generate possible spelling corrections for word."
	return (known([word]) or known(edits1(word)) or known(edits2(word)) or set([word]))


def known(words):
	# "The subset of `words` that appear in the dictionary of WORDS."
	return set(w for w in words if w in WORDS)


def edits1(word):
	# "All edits that are one edit away from `word`."
	letters = 'abcdefghijklmnopqrstuvwxyz'
	splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
	deletes = [L + R[1:] for L, R in splits if R]
	transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
	replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
	inserts = [L + c + R for L, R in splits for c in letters]
	return set(deletes + transposes + replaces + inserts)


def edits2(word):
	# "All edits that are two edits away from `word`."
	return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def corrected_term(term):
	temp = term.lower().split()
	temp = [correction(word) for word in temp]
	return ' '.join(temp)


def standardize_units(text):
	text = " " + text + " "
	text = re.sub('( gal | gals | galon )', ' gallon ', text)
	text = re.sub('( ft | fts | feets | foot | foots )', ' feet ', text)
	text = re.sub('( squares | sq )', ' square ', text)
	text = re.sub('( lb | lbs | pounds )', ' pound ', text)
	text = re.sub('( oz | ozs | ounces | ounc )', ' ounce ', text)
	text = re.sub('( yds | yd | yards )', ' yard ', text)
	return text


def preprocessing(sent):
	sent = sent.replace('in.', ' inch ')  # If we dont to this then 'in.' will be turned to 'in' in the next step
	words = re.split(r'\W+', sent)
	words = [word.lower() for word in words]
	res = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ",
				 ' '.join(words))  # add space between number and alphabets in a string
	cleaned = standardize_units(res)
	cleaned = ' '.join(cleaned.split())  # removing extra whitespaces
	return cleaned


def preprocessing_search(sent):
	sent = sent.replace('in.', ' inch ')
	words = re.split(r'\W+', sent)
	words = [word.lower() for word in words]
	res = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ",
				 ' '.join(words))  # add space between number and alphabets in a string
	res = standardize_units(res)
	res = res.replace(' in ',
					  ' inch ')  # in search_terms 'in' is used more for 'inch' than as a preposition hence this step shouldn't hurt
	cleaned = ' '.join(res.split())  # removing extra whitespaces
	return cleaned


# stop word removal and stemming
# We didn't do this before because we wanted to fix the typos in the searh term first
porter = PorterStemmer()
stp_wrds = set(stopwords.words('english'))


def futher_preprocessing(sent):
	sent = sent.replace('_', ' _ ')
	words = sent.split()
	words = [w for w in words if not w in stp_wrds]
	# words = [porter.stem(word) for word in words]
	return ' '.join(words)


# stop word removal only - no stemming
def futher_preprocessing_without_stem(sent):
	sent = sent.replace('_', ' _ ')
	words = sent.split()
	words = [w for w in words if not w in stp_wrds]
	return ' '.join(words)


def get_TFIDF_vec(df, row_name, svd_model):
	with open('./model/vectorizer.pkl', 'rb') as f:
		unpickler = pickle.Unpickler(f)
		# if file is not empty scores will be equal
		# to the value unpickled
		vectorizer = unpickler.load()
		np_arr = vectorizer.transform(df[row_name])
		return svd_model.transform(np_arr)


def count_number_in_search_attr(sentence1, sentence2):
	search_term_tokens = sentence1.split(' ')
	numbers_in_search_term = set([i for i in search_term_tokens if i.isdigit()])
	attr_tokens = sentence2.split(' ')
	numbers_attr = set([i for i in attr_tokens if i.isdigit()])
	return len(numbers_in_search_term & numbers_attr)


def common_word_count_ngrams(sentence1, sentence2, ngram=1, edit_distance_thresshold=0):
	sentence1 = sentence1.split(' ')
	sentence2 = sentence2.split(' ')
	ngrams_1 = set(ngrams(sentence1, ngram))
	ngrams_2 = set(ngrams(sentence2, ngram))
	if edit_distance_thresshold == 0:
		return len(ngrams_1 & ngrams_2)
	else:
		count = 0
		for i in ngrams_1:
			for j in ngrams_2:
				if edit_distance(i, j) <= edit_distance_thresshold:
					count += 1
		return count


# with open('./model/features_1.pkl','rb') as f:
#   features_1 = pickle.load(f)

def cosine_similarity_sent(sentence1, sentence2):
	A = set(sentence1.split())
	B = set(sentence2.split())
	numerator = len(A & B)
	denominator = math.sqrt(len(A)) * math.sqrt(len(B))

	if not denominator:
		return 0.0
	else:
		return numerator / denominator


def includes_brand(search_term, brand, edit_dis_thresshold=1):
	search_term = search_term.split(' ')
	brand = brand.split(' ')
	count = 0
	for i in search_term:
		for j in brand:
			if edit_distance(i, j) < edit_dis_thresshold:
				count += 1
	return count


def minimum_jaccard_coefficient(sentence1, sentence2):
	sentence1 = sentence1.strip().split(' ')
	sentence2 = sentence2.strip().split(' ')
	minimum_distance = 99999
	for i in sentence1:
		for j in sentence2:
			minimum_distance = min(jaccard_distance(set(i), set(j)), minimum_distance)
	return minimum_distance


def minimum_edit_distance(sentence1, sentence2):
	sentence1 = sentence1.split(' ')
	sentence2 = sentence2.split(' ')
	minimum_distance = 99999
	for i in sentence1:
		for j in sentence2:
			minimum_distance = min(edit_distance(i, j), minimum_distance)
	return minimum_distance


def get_mean_jcc_searchterm_and_title(sentence1, sentence2):
	search_term_word = sentence1.strip().split(' ')
	desc_word = sentence2.strip().split(' ')
	mean = 0.0
	for i in search_term_word:
		if i == '':
			continue
		min_ = 99999
		for j in desc_word:
			if j == '':
				continue
			min_ = min(min_, jaccard_distance(set(i), set(j)))
		mean += min_
	mean = mean / len(search_term_word)
	return mean


def sum_jcc(sentence1, sentence2):
	return jaccard_distance(set(sentence1.strip().split(' ')), set(sentence2.strip().split(' ')))


def sum_edit_distance(sentence1, sentence2):
	return edit_distance(sentence1, sentence2)


def get_mean_jcc_searchterm_and_desc(sentence1, sentence2):
	search_term_word = sentence1.strip().split(' ')
	desc_word = sentence2.strip().split(' ')
	mean = 0.0
	for i in search_term_word:
		if i == '':
			continue
		min_ = 99999
		for j in desc_word:
			if j == '':
				continue
			min_ = min(min_, jaccard_distance(set(i), set(j)))
		mean += min_
	mean = mean / len(search_term_word)
	return mean


def get_mean_jcc_searchterm_and_attr(sentence1, sentence2):
	search_term_word = sentence1.strip().split(' ')
	desc_word = sentence2.strip().split(' ')
	mean = 0.0
	for i in search_term_word:
		if i == '':
			continue
		min_ = 99999
		for j in desc_word:
			if j == '':
				continue
			min_ = min(min_, jaccard_distance(set(i), set(j)))
		mean += min_
	mean = mean / len(search_term_word)
	return mean


# with open('./model/features_2.pkl','rb') as f:
#   features_2 = pickle.load(f)

# from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(smooth_idf=True, min_df=2, max_features=10000, stop_words='english',lowercase=True)
# svd_model2 = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=10, random_state=122)
# # transformed_pt_pd_unique = vectorizer.transform(title_desc_unique)
# # svd_model2.fit(transformed_pt_pd_unique)

def okapi_bm25_fit(corpus):
	tfidf_model = TfidfVectorizer(smooth_idf=False, token_pattern=r"(?u)\b\w+\b")
	tfidf_model.fit(corpus)
	idf_dict = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))
	avgdl = np.mean([len(doc.split()) for doc in corpus])
	params = {'idf_dict': idf_dict,
			  'avgdl': avgdl,
			  'N': len(corpus)}
	return params


def okapi_bm25_score(query, doc, params, k=1.2, b=0.75):
	idf_dict = params['idf_dict']
	avgdl = params['avgdl']
	N = params['N']
	score_query = 0

	for word in query.split():
		dl = len(doc.split())
		tf = doc.count(word)
		if word in idf_dict.keys():
			idf = idf_dict[word]
		else:
			idf = np.log(N + 1)

		score_word = idf * (tf * (k + 1)) / (tf + k * (1 - b) + b * dl / avgdl)
		score_query += score_word

	return score_query


with open('./model/bm25_params_title.pkl', 'rb') as f:
	params_title_bm25 = pickle.load(f)
with open('./model/bm25_params_desc.pkl', 'rb') as f:
	params_desc_bm25 = pickle.load(f)
with open('./model/bm25_params_brand.pkl', 'rb') as f:
	params_brand_bm25 = pickle.load(f)

# with open('./model/features_5.pkl','rb') as f:
#   F3_SmM_params_title = pickle.load(f)

# F3_SmM_params_title = F3_SmM_params_title.iloc[:,8:]
#
# F3_SmM_params_title.columns

# """**Modelling**
#
# **Base Model**
# """

# Loading the standard scalers
M1_scaler_ridge = pickle.load(open('./model/M1_scaler_ridge.pkl', 'rb'))
M1_scaler_lasso = pickle.load(open('./model/M1_scaler_lasso.pkl', 'rb'))
M1_scaler_en = pickle.load(open('./model/M1_scaler_en.pkl', 'rb'))

# Loading the models
M1_xgb = pickle.load(open('./model/M1_xgb.pkl', 'rb'))
M1_rf = pickle.load(open('./model/M1_rf.pkl', 'rb'))
M1_ridge = pickle.load(open('./model/M1_ridge.pkl', 'rb'))
M1_lasso = pickle.load(open('./model/M1_lasso.pkl', 'rb'))
M1_en = pickle.load(open('./model/M1_en.pkl', 'rb'))
M1_dt = pickle.load(open('./model/M1_dt.pkl', 'rb'))

# """**Meta Model**"""

final_scaler = pickle.load(open('./model/meta_scaler.pkl', "rb"))
final_ridge = pickle.load(open('./model/meta_ridge.pkl', "rb"))


def get_search_relevance(test_set):
	test_set['cleaned_title'] = test_set['product_title'].apply(lambda x: preprocessing(x))
	test_set['cleaned_brand'] = test_set['brand'].apply(lambda x: preprocessing(x))
	test_set['cleaned_description'] = test_set['product_description'].apply(lambda x: preprocessing(x))
	test_set['cleaned_attributes'] = test_set['combined_attr'].apply(lambda x: preprocessing(x))
	test_set['cleaned_search'] = test_set['search_term'].apply(lambda x: preprocessing_search(x))

	test_set['corrected_search'] = test_set['cleaned_search'].apply(lambda x: corrected_term(x))

	cleaned_test_set = pd.DataFrame()
	cleaned_test_set['title'] = test_set['cleaned_title'].apply(lambda x: futher_preprocessing(x))
	cleaned_test_set['brand'] = test_set['cleaned_brand'].apply(lambda x: futher_preprocessing(x))
	cleaned_test_set['description'] = test_set['cleaned_description'].apply(lambda x: futher_preprocessing(x))
	cleaned_test_set['attributes'] = test_set['cleaned_attributes'].apply(lambda x: futher_preprocessing(x))
	cleaned_test_set['search'] = test_set['cleaned_search'].apply(lambda x: futher_preprocessing(x))
	cleaned_test_set['corrected_search'] = test_set['corrected_search'].apply(lambda x: futher_preprocessing(x))

	cleaned_test_set['brand'] = cleaned_test_set['brand'].replace(to_replace=[""], value="missing_brand")

	cleaned_test_set['search'] = cleaned_test_set['search'].replace(to_replace=[""], value="missing_search")

	cleaned_test_set['attributes'] = cleaned_test_set['attributes'].apply(lambda x: re.sub('bullet \d\d ', '', x))

	cleaned_test_set['description'] = cleaned_test_set['description'].apply(lambda x: re.sub('bullet \d\d ', '', x))

	cleaned_test_set.rename(columns={"search": "raw_search"}, inplace=True)
	# cleaned_test_set2.rename(columns={"search": "raw_search"}, inplace=True)

	# FEATURIZATION
	data1 = cleaned_test_set.copy()

	data1['search_term_length'] = data1.progress_apply(lambda x: len(x['corrected_search']), axis=1)
	data1['last_word_in_title'] = data1.progress_apply(
		lambda x: x['corrected_search'].split()[-1] in x['title'].split(), axis=1)
	data1['last_word_in_desc'] = data1.progress_apply(
		lambda x: x['corrected_search'].split()[-1] in x['description'].split(), axis=1)
	data1['have_brand_or_not'] = data1.progress_apply(lambda x: x['brand'] != 'NoBrand', axis=1)
	data1['have_attr_or_not'] = data1.progress_apply(lambda x: x['attributes'] != 'noattribute', axis=1)

	data1['numbers_in_attr'] = data1.progress_apply(
		lambda x: count_number_in_search_attr(x['corrected_search'], x['attributes']), axis=1)
	data1['numbers_in_title'] = data1.progress_apply(
		lambda x: count_number_in_search_attr(x['corrected_search'], x['title']), axis=1)
	data1['numbers_in_desc'] = data1.progress_apply(
		lambda x: count_number_in_search_attr(x['corrected_search'], x['description']), axis=1)

	data1['title_common_word_count'] = data1.progress_apply(
		lambda x: common_word_count_ngrams(x['corrected_search'], x['title'], edit_distance_thresshold=1), axis=1)
	data1['desc_common_word_count'] = data1.progress_apply(
		lambda x: common_word_count_ngrams(x['corrected_search'], x['description'], edit_distance_thresshold=1), axis=1)
	data1['attr_common_words_count'] = data1.progress_apply(
		lambda x: common_word_count_ngrams(x['corrected_search'], x['attributes'], edit_distance_thresshold=1), axis=1)

	data1['title_common_word_count_bigram'] = data1.progress_apply(
		lambda x: common_word_count_ngrams(x['corrected_search'], x['title'], ngram=2, edit_distance_thresshold=4),
		axis=1)
	data1['desc_common_word_count_bigram'] = data1.progress_apply(
		lambda x: common_word_count_ngrams(x['corrected_search'], x['description'], ngram=2,
										   edit_distance_thresshold=4), axis=1)
	data1['attr_common_word_count_bigram'] = data1.progress_apply(
		lambda x: common_word_count_ngrams(x['corrected_search'], x['attributes'], ngram=2, edit_distance_thresshold=4),
		axis=1)

	basic_feats = ['search_term_length', 'last_word_in_title', 'last_word_in_desc', 'have_brand_or_not',
				   'have_attr_or_not', 'numbers_in_attr', 'numbers_in_title', 'numbers_in_desc',
				   'title_common_word_count', 'desc_common_word_count', 'attr_common_words_count',
				   'title_common_word_count_bigram', 'desc_common_word_count_bigram', 'attr_common_word_count_bigram'
				   ]
	for col in basic_feats:
		data1[col] = data1[col].astype(int)
	data1 = data1.iloc[:, 6:]
	# print("DATA1-->",data1.columns)
	# Featurization part 2

	data2 = cleaned_test_set.copy()
	data2['cosine_search_term_Title'] = data2.progress_apply(
		lambda row: cosine_similarity_sent(row['corrected_search'], row['title']), axis=1)
	data2['cosine_search_term_desc'] = data2.progress_apply(
		lambda row: cosine_similarity_sent(row['corrected_search'], row['description']), axis=1)

	data2['search_term_contains_brand'] = data2.progress_apply(
		lambda x: includes_brand(x['corrected_search'], x['brand']), axis=1)

	data2['min_jcc_brand_with_search_term'] = data2.progress_apply(
		lambda x: minimum_jaccard_coefficient(x['corrected_search'], x['brand']), axis=1)
	data2['min_edit_brand_with_search_term'] = data2.progress_apply(
		lambda x: minimum_edit_distance(x['corrected_search'], x['brand']), axis=1)

	data2['min_jcc_title_with_search_term'] = data2.progress_apply(
		lambda x: minimum_jaccard_coefficient(x['corrected_search'], x['title']), axis=1)
	data2['min_edit_title_with_search_term'] = data2.progress_apply(
		lambda x: minimum_edit_distance(x['corrected_search'], x['title']), axis=1)

	data2['mean_jcc_product_title_with_search_term'] = data2.progress_apply(
		lambda x: get_mean_jcc_searchterm_and_title(x['corrected_search'], x['title']), axis=1)
	data2['mean_jcc_product_desc_with_search_term'] = data2.progress_apply(
		lambda x: get_mean_jcc_searchterm_and_desc(x['corrected_search'], x['description']), axis=1)

	data2['mean_jcc_attr_with_search_term'] = data2.progress_apply(
		lambda x: get_mean_jcc_searchterm_and_attr(x['corrected_search'], x['attributes']), axis=1)

	data2['sum_jaccard_product_title_with_search_term'] = data2.progress_apply(
		lambda x: sum_jcc(x['corrected_search'], x['title']), axis=1)
	data2['sum_edit_product_title_with_search_term'] = data2.progress_apply(
		lambda x: sum_edit_distance(x['corrected_search'], x['title']), axis=1)
	data2['sum_jcc_prod_desc_with_search_term'] = data2.progress_apply(
		lambda x: sum_jcc(x['corrected_search'], x['description']), axis=1)

	data2 = data2[['cosine_search_term_Title', 'cosine_search_term_desc', 'search_term_contains_brand',
				   'min_jcc_brand_with_search_term',
				   'min_edit_brand_with_search_term', 'min_jcc_title_with_search_term',
				   'min_edit_title_with_search_term', 'mean_jcc_product_title_with_search_term',
				   'sum_jaccard_product_title_with_search_term', 'sum_edit_product_title_with_search_term',
				   'sum_jcc_prod_desc_with_search_term',
				   'mean_jcc_product_desc_with_search_term',
				   'mean_jcc_attr_with_search_term'
				   ]]
	# print("DATA2-->",data2.columns)

	# Featurization part 3
	data3 = cleaned_test_set.copy()
	svd_model = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=10, random_state=122)
	vectorizer = TfidfVectorizer(smooth_idf=True, min_df=2, max_features=10000, stop_words='english', lowercase=True)

	title_description = data3["title"].astype(str) + '. ' + data3["description"].astype(str)

	vectorizer.fit(title_description)
	X_title_desc = vectorizer.transform(title_description)
	# print("Title Description Vector:",X_title_desc.shape)
	svd_model.fit(X_title_desc)
	truncated_title_desc = svd_model.transform(X_title_desc)
	# print("Title Description SVD:",truncated_title_desc.shape)

	title_search = data3["description"].astype(str) + '. ' + data3["corrected_search"].astype(str)

	# print(title_search)
	vectorizer.fit(title_search)
	X_search = vectorizer.transform(title_search)
	# print("Search Vector:",X_search.shape)
	svd_model.fit(X_search)
	truncated_search = svd_model.transform(X_search)
	# print("Search SVD:",truncated_search.shape)

	cosine_similarity_tfidf_search_title_des = [cosine_similarity([x], [y]) for x, y in
												zip(truncated_search, truncated_title_desc)]
	data3['cosine_similarity_tfidf_search_title_des'] = [i[0][0] for i in cosine_similarity_tfidf_search_title_des]

	# truncated_pt_train = svd_model.transform(vectorizer.transform(data3["title"].astype(str)))
	# print(truncated_pt_train.shape)

	# cosine_similarity_tfidf_search_title = [cosine_similarity([x], [y]) for x,y in zip(transformed_search, truncated_pt_train)]
	# data3['cosine_similarity_tfidf_search_title'] = [i[0][0] for i in cosine_similarity_tfidf_search_title]

	truncated_pd_train = svd_model.transform(vectorizer.transform(data3["description"].astype(str)))

	# Calculate the cosine similarity of each pair of search_term and description (sentence level) after truncated
	cosine_similarity_tfidf_search_des = [cosine_similarity([x], [y]) for x, y in
										  zip(truncated_search, truncated_pd_train)]
	data3['cosine_similarity_tfidf_search_des'] = [i[0][0] for i in cosine_similarity_tfidf_search_des]

	data3 = data3[['cosine_similarity_tfidf_search_title_des', 'cosine_similarity_tfidf_search_des']]

	# print("DATA3-->",data3.columns)

	# Featurization part 4
	data4 = cleaned_test_set.copy()
	data4['bm25_ST'] = data4.apply(
		lambda row: okapi_bm25_score(row['corrected_search'], row['title'], params_title_bm25), axis=1)
	data4['bm25_SD'] = data4.apply(
		lambda row: okapi_bm25_score(row['corrected_search'], row['description'], params_desc_bm25), axis=1)
	data4['bm25_SB'] = data4.apply(
		lambda row: okapi_bm25_score(row['corrected_search'], row['brand'], params_brand_bm25), axis=1)

	title_description = data4["title"].astype(str) + '. ' + data4["description"].astype(str)
	title_desc_unique = title_description.unique()

	idf_dict = dict(zip(vectorizer.get_feature_names(), list(vectorizer.idf_)))
	N = len(title_desc_unique)
	params = {'idf_dict': idf_dict, 'N': N}

	max_tf = []
	max_idf = []
	max_tfidf = []

	min_tf = []
	min_idf = []
	min_tfidf = []

	sum_tf = []
	sum_idf = []
	sum_tfidf = []

	for ind, row in cleaned_test_set.iterrows():
		search = row['corrected_search']
		text = row['title']
		tf_vals = []
		idf_vals = []
		tfidf_vals = []
		for word in search.split():
			if word in idf_dict.keys():
				tf = text.count(word)
				idf = idf_dict[word]
			else:
				tf = text.count(word)
				idf = np.log(N + 1)

			tf_vals.append(tf)
			idf_vals.append(idf)
			tfidf_vals.append(tf * idf)

		max_tf.append(max(tf_vals))
		min_tf.append(min(tf_vals))
		sum_tf.append(sum(tf_vals))

		max_idf.append(max(idf_vals))
		min_idf.append(min(idf_vals))
		sum_idf.append(sum(idf_vals))

		max_tfidf.append(max(tfidf_vals))
		min_tfidf.append(min(tfidf_vals))
		sum_tfidf.append(sum(tfidf_vals))

	data4['max_tf_ST'] = max_tf
	data4['max_idf_ST'] = max_idf
	data4['max_tfidf_ST'] = max_tfidf

	data4['min_tf_ST'] = min_tf
	data4['min_idf_ST'] = min_idf
	data4['min_tfidf_ST'] = min_tfidf

	data4['sum_tf_ST'] = sum_tf
	data4['sum_idf_ST'] = sum_idf
	data4['sum_tfidf_ST'] = sum_tfidf

	data4 = data4.iloc[:, 6:]

	# print("DATA4-->",data4.columns)

	# print("FINAL DATA SHAPE-->",data1.shape, data2.shape,data3.shape,data4.shape)
	# MODELLING
	X1 = pd.concat([data1, data2, data3, data4], axis=1)
	# print("FINAL_DATA-->",X1.columns)
	# print("FINAL_SHAPE-->",X1.shape)
	pred_xgb = M1_xgb.predict(X1)
	pred_rf = M1_rf.predict(X1)
	pred_ridge = M1_ridge.predict(M1_scaler_ridge.transform(X1))
	pred_lasso = M1_lasso.predict(M1_scaler_lasso.transform(X1))
	pred_en = M1_en.predict(M1_scaler_en.transform(X1))
	pred_dt = M1_dt.predict(X1)

	arr = np.hstack((
		pred_xgb.reshape(-1, 1),
		pred_rf.reshape(-1, 1),
		pred_dt.reshape(-1, 1),
		pred_ridge.reshape(-1, 1),
		pred_lasso.reshape(-1, 1),
		pred_en.reshape(-1, 1)))
	M1_df = pd.DataFrame(arr, columns=['M1_xgb', 'M1_rf', 'M1_dt', 'M1_ridge', 'M1_lasso', 'M1_en'], index=X1.index)
	# print(M1_df.shape)
	# FINAL
	test_x = M1_df
	test_x_std = final_scaler.transform(test_x)
	test_y_pred = final_ridge.predict(test_x_std)
	return test_y_pred


def get_candidates(search, N):
	search = preprocessing_search(search)
	search = corrected_term(search)
	tokenized_query = search.split(" ")
	candidates = bm25_model.get_top_n(tokenized_query, product_text, n=N)
	candidate_products = database[database['text'].isin(candidates)].drop('text', axis=1)
	candidate_products['search_term'] = search
	reorder_cols = ['product_uid', 'product_title', 'search_term', 'combined_attr', 'brand', 'product_description']
	# print(candidate_products[reorder_cols])
	return candidate_products[reorder_cols]


def main(srch, n):
	test_set = get_candidates(srch, 100)
	test_set = pd.DataFrame(test_set)
	test_set['relevance'] = get_search_relevance(test_set)
	# print(test_set.sort_values('relevance', ascending=False).head(10)[['product_title', 'relevance']])
	return test_set.sort_values('relevance', ascending=False).head(10)[['product_title', 'relevance']]
st.header("Enter your search query")												# Set the header of the page
with st.form("Search Prediction "):											# Create a form with name "SQLi Detection"
		query = st.text_input("Enter search query here")		# Create a text input with name "Enter SQL query here"
		if st.form_submit_button("Submit"):							# if the user clicks the submit button
			isSQLi = (main(query, 10))								# Call the predict_class function and store the result in isSQLi
			st.write("Your query is:", query)
			st.dataframe(isSQLi)
