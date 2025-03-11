# Deep Learning to Natural Language Processing applications - Part 1

## Main resources for NLP

### Tokenization

The process of breaking a text into words, phrases, symbols, or other meaningful elements is called tokenization. The tokens are the basic units of the text, which can be words, numbers, or punctuation marks.

### Normalization

The process of converting a text into a standard form is called normalization. This process can include converting all letters to lowercase, removing punctuation marks, and removing stop words.

### Stop words removal

Stop words are common words that do not carry much meaning, such as "the", "and", "is", "in", etc. Removing stop words can help reduce the size of the vocabulary and improve the performance of the model.

### Text representation

Text representation is the process of converting a text into a numerical form that can be used as input to a machine learning model. There are several techniques for text representation, such as bag of words, TF-IDF, and word embeddings.

## From One-Hot Encoding to Word Embeddings

### One-Hot Encoding

One-hot encoding is a technique used to represent categorical variables as binary vectors. Each category is represented by a binary vector with a 1 in the position corresponding to the category and 0s in all other positions.

### TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is a technique used to represent text documents as numerical vectors. It takes into account the frequency of a term in a document and the frequency of the term in the entire corpus.

### Word Embeddings

Word embeddings are dense vector representations of words that capture semantic relationships between words. They are learned from large text corpora using neural network models such as Word2Vec, GloVe, and FastText.

## Introduction of Word2Vec, GloVe, and FastText

### Word2Vec

Word2Vec is a neural network model that learns word embeddings from large text corpora. It uses a skip-gram or continuous bag of words (CBOW) architecture to predict the context words of a target word.

#### CBOW (Continuous Bag of Words)

The CBOW model predicts the target word based on the context words surrounding it. It is trained to maximize the probability of predicting the target word given the context words.

#### Skip-gram

The skip-gram model predicts the context words of a target word. It is trained to maximize the probability of predicting the context words given the target word.

### GloVe

GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm for obtaining word embeddings. It combines the advantages of global matrix factorization and local context window methods to learn word vectors.

### FastText

FastText is a word embedding model that extends Word2Vec by considering subword information. It represents words as bags of character n-grams and learns embeddings for subwords as well as words.

## Recurrent Neural Networks (RNNs) on NLP

### How do the RNNs work?

#### Sequential input

RNNs are designed to handle sequential input data, such as text or time series data. They process the input data one element at a time, updating their internal state at each step.

#### Hidden state

RNNs maintain a hidden state that captures information about the sequence seen so far. The hidden state is updated at each time step based on the input and the previous hidden state.

#### Output

RNNs can produce an output at each time step or only at the final time step. The output can be used for tasks such as sequence classification, sequence generation, or sequence-to-sequence mapping.

### NLP Applications

- Language modeling
- Automatic translation
- Speech recognition
- Text generation
- Sentiment analysis
