import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

class TextPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000)
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
        tokens = self.tokenizer.tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(lemmatized_words)


    def fit_transform(self, X):
        cleaned_text = [self.clean_text(text) for text in X]
        return self.vectorizer.fit_transform(cleaned_text)

    def transform(self, X):
        cleaned_text = [self.clean_text(text) for text in X]
        return self.vectorizer.transform(cleaned_text)
