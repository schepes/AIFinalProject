from utils.data_loader import DataLoader
from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.bert_preprocessor import BertPreprocessor
from models.baseline_model import BaselineModel
from models.bert_model import BertModel
import pandas as pd
import torch
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

def main():
    # Load the data
    loader = DataLoader('data/')
    df = loader.load_data()

    # Preprocess the text data for baseline model
    preprocessor = TextPreprocessor()
    df['processed_text'] = df['statement'].apply(preprocessor.clean_text)
    X_vectorized = preprocessor.fit_transform(df['processed_text'])

    # Extract the label for baseline model
    y_baseline = df['label'].map({'true': 1, 'half-true': 1, 'mostly-true': 1, 'false': 0, 'pants-fire': 0, 'barely-true': 0})

    # Train and evaluate the baseline model
    baseline_model = BaselineModel(X_vectorized, y_baseline)
    baseline_model.train_and_evaluate()



if __name__ == "__main__":
    main()
