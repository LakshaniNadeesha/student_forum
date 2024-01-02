from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import bigrams, FreqDist
from nltk.corpus import stopwords
from nltk import ne_chunk
import pickle, nltk

app = Flask(__name__, template_folder='templates')

def load_models():
    # Load the Random Forest model and Word2Vec model from the saved file
    with open('rf_model.pickle', 'rb') as file:
        rf_model, word2vec_model = pickle.load(file)
    return rf_model, word2vec_model

def average_word2vec(tokens, model, vector_size):
    vector_sum = sum(model.wv[word] for word in tokens if word in model.wv)
    return vector_sum / len(tokens) if len(tokens) > 0 else np.zeros(vector_size)

def preprocess_text(text):
    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    # Remove punctuation and other non-alphabetic characters
    tokens = [word for word in tokens if word.isalpha()]
    return tokens

def extract_named_entities(text):
    tokens = preprocess_text(text)
    tagged_tokens = nltk.pos_tag(tokens)
    named_entities = ne_chunk(tagged_tokens)
    return named_entities

def count_named_entities(text):
    named_entities = extract_named_entities(text)
    count = sum(1 for chunk in named_entities if hasattr(chunk, 'label'))
    return count

def compute_cohesion(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    # Tokenize each sentence into words
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

    # Flatten the list of lists
    words = [word for sentence in tokenized_sentences for word in sentence if word.isalnum() and word not in stop_words]

    # Compute word overlap between consecutive sentences
    word_overlap_count = 0
    for i in range(len(tokenized_sentences) - 1):
        sentence1 = set(tokenized_sentences[i])
        sentence2 = set(tokenized_sentences[i + 1])
        word_overlap_count += len(sentence1.intersection(sentence2))

    # Compute cohesion as the total word overlap normalized by the total number of words
    total_words = len(words)
    cohesion = word_overlap_count / total_words if total_words > 0 else 0

    return cohesion

def calculate_coherence(text):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = [word.lower() for sentence in sentences for word in word_tokenize(sentence)]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Calculate bigrams
    bi_grams = list(bigrams(words))

    # Calculate frequency distribution of words and bigrams
    freq_dist_words = FreqDist(words)
    freq_dist_bigrams = FreqDist(bi_grams)

    # Calculate Pointwise Mutual Information (PMI)
    coherence = sum([freq_dist_bigrams[bigram] * freq_dist_words[bigram[0]] * freq_dist_words[bigram[1]] for bigram in bi_grams])

    return coherence

def word_count(text):
    # Use split() to break the text into words
    words = text.split()

    # Count the number of words
    count = len(words)

    return count

def preprocess_input_for_prediction(parent, student, created, subject,message, word2vec_model):
     # Create a DataFrame with user inputs
    data = pd.DataFrame({'parent': [parent],
                         'student': [student],
                         'created': [created],
                         'subject': [subject],
                         'message': [message]})

    # Preprocess the 'created' column
    data['created'] = pd.to_datetime(data['created'])
    data['hour'] = data['created'].dt.hour

    # Additional feature computations
    data['named_entities_count'] = data['message'].apply(count_named_entities)
    data['cohesion'] = data['message'].apply(compute_cohesion)
    data['coherence'] = data['message'].apply(calculate_coherence)
    data['wc'] = data['message'].apply(word_count)

    # Combine text features
    data['combined_text'] = data[['parent', 'student', 'subject', 'message']].astype(str).agg(' '.join, axis=1)

    # Tokenize the text
    data['tokenized_text'] = data['combined_text'].apply(lambda x: word_tokenize(x.lower()))

    # Create Word2Vec vectors for each text using the provided model
    data['word2vec'] = data['tokenized_text'].apply(lambda x: average_word2vec(x, word2vec_model, vector_size=100))

    # Convert Word2Vec vectors to DataFrame columns
    word2vec_columns = pd.DataFrame(data['word2vec'].to_list(), columns=[f'w2v_{i}' for i in range(100)])

    # Combine Word2Vec features, hour, and other features
    X_combined = pd.concat([word2vec_columns, data[['hour', 'named_entities_count', 'cohesion', 'coherence', 'wc']]], axis=1)

    return X_combined

def predict_label_with_word2vec(parent, student, created, subject, message, rf_model, word2vec_model):
    # Preprocess the input for prediction
    input_data = preprocess_input_for_prediction(parent, student, created, subject, message, word2vec_model)

    # Make predictions using the pre-trained Random Forest model
    prediction = rf_model.predict(input_data)

    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_parent = request.form['parent']
    user_student = request.form['student']
    user_created = request.form['created']
    user_subject = request.form['subject']
    user_message = request.form['message']

    rf_model, word2vec_model = load_models()

    prediction = predict_label_with_word2vec(user_parent, user_student, user_created, user_subject, user_message, rf_model, word2vec_model)

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
