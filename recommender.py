import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import string
from typing import List, Tuple, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data (if not already present)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class ContentRecommender:
    """
    A content-based recommendation system that uses TF-IDF vectorization
    and cosine similarity to recommend items based on textual content.
    """

    def __init__(self, data_path: str = "sample_data.csv"):
        """
        Initialize the recommender with data from a CSV file.
        
        Args:
            data_path (str): Path to the CSV file containing item data
        """
        self.data_path = data_path
        self.df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        """
        Load data from CSV and prepare it for recommendation processing.
        """
        self.df = pd.read_csv(self.data_path)
        # Combine text fields for richer feature representation
        self.df['combined_features'] = (
            self.df['title'].fillna('') + ' ' +
            self.df['description'].fillna('') + ' ' +
            self.df['tags'].fillna('') + ' ' +
            self.df['category'].fillna('')
        )
        self._preprocess_text()
        self._create_tfidf_matrix()

    def _preprocess_text(self):
        """
        Preprocess text data with advanced NLP techniques.
        """
        def clean_and_tokenize(text):
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation and special characters
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            # Tokenize
            tokens = word_tokenize(text)
            # Remove stopwords and stem
            tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
            # Join back to string
            return ' '.join(tokens)
        
        self.df['cleaned_features'] = self.df['combined_features'].apply(clean_and_tokenize)

    def _create_tfidf_matrix(self):
        """
        Create TF-IDF matrix from cleaned text features.
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=1,
            max_df=0.8
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['cleaned_features'])

    def _preprocess_user_preferences(self, preferences: str) -> str:
        """
        Preprocess user preferences text in the same way as item features.
        
        Args:
            preferences (str): Raw user preferences text
            
        Returns:
            str: Cleaned preferences text
        """
        # Convert to lowercase
        preferences = preferences.lower()
        # Remove punctuation and special characters
        preferences = re.sub(r'[^a-zA-Z0-9\s]', '', preferences)
        # Tokenize
        tokens = word_tokenize(preferences)
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        # Join back to string
        return ' '.join(tokens)

    def recommend_by_text(self, preferences: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend items based on user preferences using cosine similarity.
        
        Args:
            preferences (str): User's text preferences
            top_k (int): Number of top recommendations to return
            
        Returns:
            List[Dict]: List of recommended items with scores and explanations
        """
        # Preprocess user preferences
        cleaned_preferences = self._preprocess_user_preferences(preferences)
        
        # Transform user preferences to TF-IDF vector
        user_tfidf = self.tfidf_vectorizer.transform([cleaned_preferences])
        
        # Calculate cosine similarities
        cosine_similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        
        # Get top-k similar items
        top_indices = cosine_similarities.argsort()[-top_k:][::-1]
        
        # Prepare results with explanations
        recommendations = []
        for idx in top_indices:
            item = self.df.iloc[idx]
            score = cosine_similarities[idx]
            
            # Generate explanation based on matching features
            explanation = self._generate_explanation(preferences, item)
            
            recommendations.append({
                'id': item['id'],
                'title': item['title'],
                'description': item['description'][:150] + '...' if len(item['description']) > 150 else item['description'],
                'score': round(score, 4),
                'explanation': explanation,
                'category': item['category'],
                'image_url': item.get('image_url', '')
            })
        
        return recommendations

    def _generate_explanation(self, preferences: str, item: pd.Series) -> str:
        """
        Generate an explanation for why an item was recommended.
        
        Args:
            preferences (str): User's preferences text
            item (pd.Series): Item data
            
        Returns:
            str: Explanation text
        """
        # Simple keyword matching for explanation
        pref_words = set(preferences.lower().split())
        item_words = set((item['combined_features'].lower() + ' ' + 
                         item['category'].lower()).split())
        
        common_words = pref_words.intersection(item_words)
        
        if len(common_words) > 0:
            return f"This item matches your interest in: {', '.join(list(common_words)[:3])}"
        else:
            # Fallback to category-based explanation
            return f"This {item['category'].lower()} is recommended based on content similarity to your preferences."

    def get_similar_items(self, item_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find items similar to a given item by ID.
        
        Args:
            item_id (int): ID of the item to find similar items for
            top_k (int): Number of similar items to return
            
        Returns:
            List[Dict]: List of similar items
        """
        # Find the item index
        item_idx = self.df[self.df['id'] == item_id].index
        if len(item_idx) == 0:
            return []
        
        item_idx = item_idx[0]
        
        # Get the item's TF-IDF vector
        item_vector = self.tfidf_matrix[item_idx]
        
        # Calculate cosine similarities with all items
        cosine_similarities = cosine_similarity(item_vector, self.tfidf_matrix).flatten()
        
        # Get top-k similar items (excluding the item itself)
        top_indices = cosine_similarities.argsort()[-(top_k+1):-1][::-1]
        
        # Prepare results
        similar_items = []
        for idx in top_indices:
            item = self.df.iloc[idx]
            score = cosine_similarities[idx]
            
            similar_items.append({
                'id': item['id'],
                'title': item['title'],
                'description': item['description'][:100] + '...' if len(item['description']) > 100 else item['description'],
                'score': round(score, 4),
                'category': item['category'],
                'image_url': item.get('image_url', '')
            })
        
        return similar_items

    def get_categories(self) -> List[str]:
        """
        Get all unique categories in the dataset.
        
        Returns:
            List[str]: List of unique categories
        """
        return self.df['category'].unique().tolist()

    def get_items_by_category(self, category: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get top items from a specific category.
        
        Args:
            category (str): Category to filter by
            top_k (int): Number of items to return
            
        Returns:
            List[Dict]: List of items in the category
        """
        # Filter by category
        category_items = self.df[self.df['category'] == category]
        
        # If we don't have enough items, return what we have
        if len(category_items) < top_k:
            top_k = len(category_items)
            
        # Return top items (first top_k items since there's no scoring)
        result = []
        for _, item in category_items.head(top_k).iterrows():
            result.append({
                'id': item['id'],
                'title': item['title'],
                'description': item['description'][:100] + '...' if len(item['description']) > 100 else item['description'],
                'category': item['category'],
                'image_url': item.get('image_url', '')
            })
        
        return result


# Create a global instance for easy access
recommender = ContentRecommender()


def recommend_by_text(preferences: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function to get recommendations based on text preferences.
    
    Args:
        preferences (str): User's text preferences
        top_k (int): Number of top recommendations to return
        
    Returns:
        List[Dict]: List of recommended items with scores and explanations
    """
    return recommender.recommend_by_text(preferences, top_k)