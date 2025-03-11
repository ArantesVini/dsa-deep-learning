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

### Challenges of RNNs

#### Long-term dependencies

RNNs have difficulty capturing long-term dependencies in sequences, as the gradients tend to vanish or explode during training. This can lead to the model forgetting information from earlier time steps.

#### Issues with vanishing / exploding gradients

The vanishing and exploding gradient problems can make it difficult to train RNNs on long sequences. Techniques such as gradient clipping, batch normalization, and using alternative architectures like LSTMs and GRUs can help mitigate these issues.

#### Sequential processing

RNNs process input sequences sequentially, which can be slow and limit parallelization. This can be a bottleneck for training and inference on large datasets.

### LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) variants

#### LSTM

LSTMs are a type of RNN architecture that includes memory cells and gating mechanisms to better capture long-term dependencies. They have separate input, forget, and output gates that control the flow of information through the cell.

#### GRU

GRUs are a simplified version of LSTMs that combine the forget and input gates into a single update gate. They have fewer parameters than LSTMs and are faster to train, but may not capture long-term dependencies as effectively.

## Attention Mechanisms and Transformers

### Attention Mechanisms

Attention mechanisms allow models to focus on different parts of the input sequence when making predictions. They assign weights to the input elements based on their relevance to the current prediction, allowing the model to attend to important information.

### Transformers

Transformers are a type of neural network architecture that relies on self-attention mechanisms to process sequences. They consist of an encoder and a decoder, each composed of multiple layers of self-attention and feedforward neural networks.

#### Self-attention

Self-attention allows each element in the input sequence to attend to all other elements, capturing dependencies between distant positions. It computes attention weights based on the similarity between the query, key, and value vectors of each element.

#### Encoder

The encoder processes the input sequence using self-attention layers to capture global dependencies and feedforward layers to model local interactions. It produces a sequence of hidden states that encode the input information.

#### Decoder

The decoder generates the output sequence based on the encoder's hidden states and the target sequence. It uses self-attention to attend to the input and output sequences and feedforward layers to predict the next token in the output sequence.

### Applications and impact

- Automatic translation
- Text summarization
- Text generation
- Text comprehension and question answering
- Text classification
- Generate like 90% of what I wrote here

## The Transformer revolution in NLP

### Transformers characteristics

- Attention mechanism
- Parallel processing
- Flexibility and generalization

### Impact on language models

- Bert: Bidirectional Encoder Representations from Transformers
- GPT: Generative Pre-trained Transformer
- Diversity of pre-trained models

## BERT (Bidirectional Encoder Representations from Transformers)

### Main features

- Bidirectional training to comprehend context for each word
- Trained in two main NLP tasks: masked language modeling and next sentence prediction
- Fine-tuning for specific tasks

### BERT applications

- Text comprehension
- Text classification
- Sentiment analysis
- Question answering

## GPT (Generative Pre-trained Transformer)

### Main features

- self-regressive training to generate continuous text with a prompt
- Able to work on NLP tasks without fine-tuning, just by adding a task-specific prompt

### GPT applications

- Generate creative text
- Email generation
- Website content creation
- Basic programming

## T5 (Text-to-Text Transfer Transformer)

### Main features

- Unified architecture for all NLP tasks
- pretraining objective - include all NLP tasks in a single model

### T5 applications

- language translation
- text summarization
- classification, and more

## Seq2Seq models and Applications

### Seq2Seq models structure

- Encoder: reads the input sequence and produces a fixed-size representation, is a RNN, LSTM, or a GRU
- Decore: generates the output sequence based on the encoder's representation, is a RNN, LSTM, or a GRU

### Applciations of Seq2Seq models

- Automatic translation
- Text summarization
- Dialogue systems/chatbots
- STT (Speech-to-Text) systems
- Grammar correction

## Zero-shot learning and LLMs

### Zero-shot learning on LLMs

- Zero-shot learning is the ability of a model to perform a task without any training examples
- Requires a vast amount of pre-training data

### Strategies to implement zero-shot learning

- Prompt engineering: design a prompt that guides the model to perform the desired task
- Using in rich contexts
- Dynamic adaptation

### Zero-shot learning applications

- Text classification
- Content generation
- Question answering
- Language translation
- Fake news detection
