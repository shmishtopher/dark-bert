# Copyright 2022 Christopher K. Schmitt
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from pathlib import Path
from bs4 import BeautifulSoup
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import nltk as nltk

# The list of huggingface transformers with tensorflow
# support and compatible tokenizers.
available_models = {
  "bert": "sentence-transformers/multi-qa-distilbert-cos-v1",
  "albert": "sentence-transformers/paraphrase-albert-small-v2",
  "roberta": "sentence-transformers/all-distilroberta-v1",
}

display_titles = {
  "bert": "BERT",
  "albert": "ALBERT",
  "roberta": "RoBERTa",
}

# Define the CLI interface for modeling our data with 
# different transformer models.  We want to control the 
# type of the tokenizer and the transformer we use, as well
# as the input and output directories
parser = ArgumentParser()
parser.add_argument("-m", "--model", choices=available_models.keys(), required=True)
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)

args = parser.parse_args()
input_dir = args.input
output_dir = args.output
model_name = available_models[args.model]
display_name = display_titles[args.model]

# To remove random glyphs and other noise, we
# only extract words in the nltk corpus
nltk.download("words")
words = set(nltk.corpus.words.words())

def extract_words(document):
  cleaned  = ""

  for word in nltk.wordpunct_tokenize(document):
    if word.lower() in words:
      cleaned += word.lower() + " "

  return cleaned

# Iterate over all of the files in the provided data 
# directory.  Parse each file with beautiful soup to parse
# the relevant text out of the markup.
data = Path(input_dir).iterdir()
data = map(lambda doc: doc.read_bytes(), data)
data = map(lambda doc: BeautifulSoup(doc, "html.parser"), data)
data = map(lambda doc: doc.get_text(), data)
data = filter(lambda doc: len(doc) > 0, data)
data = map(extract_words, data)
data = filter(lambda doc: len(doc) > 10, data)
data = list(data)

# Initilize transformer models and predict all of the
# document embeddings as computed by bert and friends
model = SentenceTransformer(model_name)
embeddings = model.encode(data, show_progress_bar=True)

# Fit TSNE model for embedding space.  Sqush down to 2
# dimentions for visualization purposes.
tsne = TSNE(n_components=2, random_state=2, init="pca", learning_rate="auto", perplexity=40)
tsne = tsne.fit_transform(embeddings)

# Hyperparameter optimizations
silhouettes = []
outliers = []
ch = []

for eps in np.arange(0.001, 1, 0.001):
  dbscan = DBSCAN(eps, metric="cosine", n_jobs=-1)
  dbscan = dbscan.fit_predict(embeddings)

  if len(np.unique(dbscan)) > 1:
    silhouettes.append(silhouette_score(embeddings, dbscan, metric="cosine"))
    ch.append(calinski_harabasz_score(embeddings, dbscan))
  else:
    silhouettes.append(0)
    ch.append(0)

  outliers.append(len(dbscan[dbscan == -1]))

for p in range(15, 51):
  best = np.argmax(silhouettes)

  dbscan = DBSCAN(0.001 + 0.001 * best, metric="cosine", n_jobs=-1)
  dbscan = dbscan.fit_predict(embeddings)

  tsne = TSNE(n_components=2, perplexity=p, learning_rate="auto", init="pca", metric="cosine")
  tsne = tsne.fit_transform(embeddings)

  plt.figure()
  plt.scatter(tsne[dbscan != -1][:, 0], tsne[dbscan != -1][:, 1], s=0.5, c=dbscan[dbscan != -1], cmap="hsv")
  plt.scatter(tsne[dbscan == -1][:, 0], tsne[dbscan == -1][:, 1], s=0.5, c="#abb8c3")
  plt.title(f"{display_name} Embeddings Visualized with T-SNE (p = {p})")
  plt.savefig(f"{output_dir}/tnse_{p:02}.png", format="png", dpi=600)
  plt.close()

plt.figure()
plt.plot(np.arange(0.001, 1, 0.001), silhouettes, lw=0.5, color="#dc322f")
plt.legend()
plt.xlabel("Epsilon")
plt.ylabel("silhouette score")
plt.title("Optimizing Epsilon by Silhouette Score")
plt.savefig(f"silhouettes.png", format="png", dpi=600)
plt.close()

plt.figure()
plt.plot(np.arange(0.001, 1, 0.001), outliers, lw=0.5, color="#dc322f")
plt.legend()
plt.xlabel("Epsilon")
plt.ylabel("outliers")
plt.title("Optimizing Epsilon by Number of Outliers")
plt.savefig(f"outliers.png", format="png", dpi=600)
plt.close()

plt.figure()
plt.plot(np.arange(0.001, 1, 0.001), ch, lw=0.5, color="#dc322f")
plt.legend()
plt.xlabel("Epsilon")
plt.ylabel("Calinski-Harabasz score")
plt.title("Optimizing Epsilon by Calinski-Harabasz Score")
plt.savefig(f"calinski-harabasz.png", format="png", dpi=600)
plt.close()