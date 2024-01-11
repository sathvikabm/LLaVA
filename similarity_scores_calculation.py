import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process similarity scores for a given dataset.')
parser.add_argument('-dataset', type=str, help='Path to the dataset file', required=True)
parser.add_argument('-output_dataset', type=str, help='Path to save the output similarity matrix', required=True)
args = parser.parse_args()

processed_data = pd.read_csv(args.dataset)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_data['preprocessed_description'])

vec_matrix = np.vstack(processed_data['vec_description'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' ')))

similarity_matrix_tfidf = cosine_similarity(tfidf_matrix)
similarity_matrix_word2vec = cosine_similarity(vec_matrix)

weight_word2vec = 0.5
weight_tfidf = 0.5

combined_similarity_matrix = (weight_word2vec * similarity_matrix_word2vec) + (weight_tfidf * similarity_matrix_tfidf)

output_path = args.output_dataset
combined_similarity_matrix_df = pd.DataFrame(combined_similarity_matrix)
combined_similarity_matrix_df.to_csv(output_path)

print(f"Combined similarity matrix saved to '{output_path}'")
