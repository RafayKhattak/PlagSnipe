# Import required libraries
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import nltk
from nltk.probability import FreqDist
import plotly.express as px
from collections import Counter
from nltk.corpus import stopwords
import string
nltk.download('punkt')
nltk.download('stopwords')


# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def calculate_perplexity(text):
    # Encode the input text using the tokenizer
    encoded_input = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
    input_ids = encoded_input[0]

    with torch.no_grad():
        # Pass the input ids through the model to get the logits
        outputs = model(input_ids)
        logits = outputs.logits

    # Calculate perplexity using cross-entropy loss
    perplexity = torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1)))
    return perplexity.item()

def calculate_burstiness(text):
    # Tokenize the text into individual words
    tokens = nltk.word_tokenize(text.lower())
    
    # Count the frequency of each word
    word_freq = FreqDist(tokens)
    
    # Count the number of words that appear more than once
    repeated_count = sum(count > 1 for count in word_freq.values())
    
    # Calculate burstiness score
    burstiness_score = repeated_count / len(word_freq)
    return burstiness_score

def plot_top_repeated_words(text):
    # Tokenize the text and remove stopwords and special characters
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]

    # Count the occurrence of each word
    word_counts = Counter(tokens)

    # Get the top 10 most repeated words
    top_words = word_counts.most_common(10)

    # Extract the words and their counts for plotting
    words = [word for word, count in top_words]
    counts = [count for word, count in top_words]

    # Plot the bar chart using Plotly
    fig = px.bar(x=words, y=counts, labels={'x': 'Words', 'y': 'Counts'}, title='Top 10 Most Repeated Words')
    st.plotly_chart(fig, use_container_width=True)
