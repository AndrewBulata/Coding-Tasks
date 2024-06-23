# Coding Tasks

# Sentiment Analysis README

## Overview

This code performs sentiment analysis on Amazon product reviews, specifically focusing on identifying and plotting the most frequent positive and negative adjectives. The code uses natural language processing (NLP) techniques to preprocess the reviews, extract adjectives, and evaluate their sentiment polarity. The final output includes histograms showing the top positive and negative adjectives.

## Prerequisites

To run this code, you need to install several Python libraries. Below is a list of the required libraries and the commands to install them using `pip`.

### Required Libraries

1. **spaCy**: A popular NLP library.
2. **spacytextblob**: A TextBlob sentiment analysis extension for spaCy.
3. **pandas**: A library for data manipulation and analysis.
4. **matplotlib**: A library for creating visualisations.
5. **collections**: Included in the Python standard library.
6. **string**: Included in the Python standard library.

### Installation Commands

Open your terminal or command prompt and enter the following commands to install the necessary modules:

```bash
pip install spacy
pip install spacytextblob
pip install pandas
pip install matplotlib
```

After installing spaCy, you need to download the spaCy model used in the script:

```bash
python -m spacy download en_core_web_sm
```

## Files

- `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`: This is the dataset containing Amazon product reviews, which is used for sentiment analysis. Ensure this file is in the same directory as the script.

## Code Explanation

### Importing Necessary Modules

The script starts by importing necessary modules:

```python
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from collections import Counter
import string
import pandas as pd
import matplotlib.pyplot as plt
```

### Loading the spaCy Model and Adding spacytextblob

The spaCy model is loaded, and spacytextblob is added for sentiment analysis:

```python
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')
```

### Loading the Dataset

The dataset containing Amazon reviews is loaded using pandas:

```python
reviews = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')
```

### Preprocessing Reviews

The `preprocess_reviews` function removes stopwords, punctuation, and converts the text to lowercase:

```python
def preprocess_reviews(reviews):
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    filtered_tokens = []

    for sentence in reviews['reviews.text']:
        doc = nlp(sentence)
        tokens = [token.text.lower() for token in doc if not token.is_stop and token.text not in string.punctuation]
        filtered_tokens.append(tokens)
    
    return filtered_tokens
```

### Filtering Adjectives

The `filter_adjectives` function retrieves the most frequent adjectives in the reviews:

```python
def filter_adjectives(tokens):
    adjectives = []
    for token in tokens:
        doc = nlp(token)
        for token in doc:
            if token.pos_ == 'ADJ':
                adjectives.append(token.text)
    return adjectives
```

### Filtering Negative Adjectives

The `filter_adjectives_with_negative_polarity` function retrieves adjectives with negative polarities:

```python
def filter_adjectives_with_negative_polarity(tokens):
    negative_adjectives = []
    for token in tokens:
        doc = nlp(token)
        for token in doc:
            if token.pos_ == 'ADJ' and doc._.polarity < 0:
                negative_adjectives.append(token.text)
    return negative_adjectives
```

### Plotting Histograms

The `plot_adjective_histograms` function plots histograms of the top positive and negative adjectives:

```python
def plot_adjective_histograms(positive_adjectives, negative_adjectives):
    pos_adj_freq = Counter(positive_adjectives)
    neg_adj_freq = Counter(negative_adjectives)

    top_positive_adjectives = pos_adj_freq.most_common(10)
    top_negative_adjectives = neg_adj_freq.most_common(10)

    pos_adjectives, pos_counts = zip(*top_positive_adjectives)
    neg_adjectives, neg_counts = zip(*top_negative_adjectives)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.bar(pos_adjectives, pos_counts, color='green')
    plt.xlabel('Positive Adjectives')
    plt.ylabel('Frequency')
    plt.title('Top Positive Adjectives')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.bar(neg_adjectives, neg_counts, color='red')
    plt.xlabel('Negative Adjectives')
    plt.ylabel('Frequency')
    plt.title('Top Negative Adjectives')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
```

### Main Function

The `main` function orchestrates the preprocessing, sentiment analysis, and plotting steps:

```python
def main():
    tokens = preprocess_reviews(reviews)
    flat_tokens = [token for sublist in tokens for token in sublist]

    positive_adjectives = filter_adjectives(flat_tokens)
    negative_adjectives = filter_adjectives_with_negative_polarity(flat_tokens)

    adj_freq_positive = Counter(positive_adjectives)
    adj_freq_negative = Counter(negative_adjectives)

    top_positive_adjectives = adj_freq_positive.most_common(10)
    top_negative_adjectives = adj_freq_negative.most_common(10)

    print("\033[1mTop Positive Adjectives:\033[0m")
    for adj, freq in top_positive_adjectives:
        print(f"{adj}: {freq}")

    print("\033[1mTop Negative Adjectives:\033[0m")
    for adj, freq in top_negative_adjectives:
        print(f"{adj}: {freq}")

    plot_adjective_histograms(positive_adjectives, negative_adjectives)

if __name__ == "__main__":
    main()
```

This script processes the reviews, performs sentiment analysis, extracts positive and negative adjectives, and plots their frequencies in histograms. Make sure all the required libraries are installed and the dataset is available to run the script successfully.

## Addtional test details
Average completion time: ~6m
Size of dataset: 5000 reviews, 806742 characters
