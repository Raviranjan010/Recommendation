import unittest
import pandas as pd
from recommender import ContentRecommender, recommend_by_text


class TestRecommender(unittest.TestCase):
    """Unit tests for the recommendation system."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.recommender = ContentRecommender("sample_data.csv")

    def test_data_loading(self):
        """Test that data loads correctly and has expected structure."""
        # Check that dataframe is not empty
        self.assertFalse(self.recommender.df.empty)
        
        # Check that required columns exist
        required_columns = ['id', 'title', 'description', 'tags', 'category']
        for col in required_columns:
            self.assertIn(col, self.recommender.df.columns)
        
        # Check that we have at least 10 items
        self.assertGreaterEqual(len(self.recommender.df), 10)

    def test_recommend_function_returns_correct_type(self):
        """Test that recommend_by_text returns the correct data type."""
        recommendations = recommend_by_text("action movies", 3)
        
        # Check that result is a list
        self.assertIsInstance(recommendations, list)
        
        # Check that list is not empty
        self.assertGreater(len(recommendations), 0)
        
        # Check that each item is a dictionary
        for item in recommendations:
            self.assertIsInstance(item, dict)

    def test_recommend_function_returns_correct_length(self):
        """Test that recommend_by_text returns the correct number of items."""
        k = 3
        recommendations = recommend_by_text("drama movies", k)
        
        # Check that we get exactly k recommendations
        self.assertEqual(len(recommendations), k)
        
        # Check that each recommendation has required keys
        required_keys = ['id', 'title', 'description', 'score', 'explanation', 'category']
        for item in recommendations:
            for key in required_keys:
                self.assertIn(key, item)

    def test_get_categories(self):
        """Test that get_categories returns the correct data."""
        categories = self.recommender.get_categories()
        
        # Check that result is a list
        self.assertIsInstance(categories, list)
        
        # Check that list is not empty
        self.assertGreater(len(categories), 0)
        
        # Check that all categories are strings
        for category in categories:
            self.assertIsInstance(category, str)

    def test_get_similar_items(self):
        """Test that get_similar_items returns the correct data structure."""
        # Test with a valid item ID
        similar_items = self.recommender.get_similar_items(1, 3)
        
        # Check that result is a list
        self.assertIsInstance(similar_items, list)
        
        # Check length (could be less than requested if not enough items)
        self.assertLessEqual(len(similar_items), 3)
        
        # If we have results, check structure
        if similar_items:
            required_keys = ['id', 'title', 'description', 'score', 'category']
            for item in similar_items:
                for key in required_keys:
                    self.assertIn(key, item)


if __name__ == '__main__':
    unittest.main()