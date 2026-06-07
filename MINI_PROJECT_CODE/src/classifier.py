import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

class ExpenseCategorizer:
    def __init__(self):
        # Create a Machine Learning pipeline:
        # 1. TF-IDF Vectorizer: Converts text descriptions to a matrix of numerical TF-IDF features.
        #    - Term Frequency (TF): How often a word appears in a description.
        #    - Inverse Document Frequency (IDF): Penalizes common words that appear across all categories.
        # 2. Multinomial Naive Bayes (MultinomialNB): Probabilistic classifier based on Bayes' Theorem,
        #    highly efficient and standard for text/document classification.
        self.model = make_pipeline(
            TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2)),
            MultinomialNB(alpha=1.0)
        )
        self.is_trained = False
        
    def train(self, expenses_list):
        """
        Trains the classifier using historical expense descriptions and categories.
        """
        if not expenses_list or len(expenses_list) < 5:
            # Not enough data to train a model, fallback to rule-based matching
            self.is_trained = False
            return False
            
        # Convert list of dictionaries to a Pandas DataFrame
        df = pd.DataFrame(expenses_list)
        
        # Extract features (Description) and target labels (Category)
        X = df['Description'].fillna('')
        y = df['Category']
        
        # Check if we have at least 2 distinct classes to train a classifier
        if len(y.unique()) < 2:
            self.is_trained = False
            return False
            
        try:
            # Fit the machine learning pipeline
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"[Warning] Failed to train classification model: {e}")
            self.is_trained = False
            return False

    def predict(self, description, fallback_category="Others"):
        """
        Predicts the category of a new expense description.
        If the model is not trained, uses a rule-based lookup.
        """
        desc_clean = description.strip().lower()
        
        # Step 1: Attempt Rule-Based Lookup (Quick Heuristics)
        # This is a solid hybrid design pattern (Heuristics + ML) used in production.
        rules = {
            "coffee": "Food", "starbucks": "Food", "burger": "Food", "pizza": "Food",
            "swiggy": "Food", "zomato": "Food", "dinner": "Food", "lunch": "Food", "grocery": "Food",
            "uber": "Travel", "ola": "Travel", "cab": "Travel", "metro": "Travel", "gas": "Travel",
            "petrol": "Travel", "flight": "Travel", "bus": "Travel",
            "netflix": "Entertainment", "spotify": "Entertainment", "movie": "Entertainment",
            "game": "Entertainment", "concert": "Entertainment", "show": "Entertainment",
            "electricity": "Utilities", "wifi": "Utilities", "broadband": "Utilities",
            "recharge": "Utilities", "water bill": "Utilities", "phone bill": "Utilities",
            "amazon": "Shopping", "nike": "Shopping", "shoes": "Shopping", "shirt": "Shopping",
            "clothes": "Shopping", "book": "Shopping", "mall": "Shopping",
            "medicine": "Others", "hospital": "Others", "haircut": "Others", "copy": "Others"
        }
        
        for keyword, category in rules.items():
            if keyword in desc_clean:
                return category
                
        # Step 2: Use Machine Learning model if trained
        if self.is_trained:
            try:
                # Predict category using the Naive Bayes model
                prediction = self.model.predict([description])
                return prediction[0]
            except Exception:
                pass
                
        # Step 3: Default fallback
        return fallback_category
