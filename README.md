# Content-Based Recommendation System

A production-ready content-based recommendation system built with Python, scikit-learn, NLTK, and Streamlit. This system recommends items based on textual content similarity using TF-IDF vectorization and cosine similarity with advanced NLP preprocessing.

## Key Features

- **Content-Based Filtering**: Recommends items based on similarity to user preferences
- **Advanced Text Processing**: Uses NLTK for tokenization, stemming, and stopword removal
- **Multiple Interfaces**: 
  - Text-based recommendations
  - Category browsing
  - Similar item finder
- **Rich UI**: Interactive Streamlit interface with item cards and visualizations
- **Export Functionality**: Download recommendations as CSV
- **Fully Deployable**: Ready for Streamlit Community Cloud or Render

## How to Run Locally in Qoder

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. The app will open in your browser. If not, navigate to `http://localhost:8501`.

## Deployment Instructions

### Deploy to Streamlit Community Cloud

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and enter your repository details
4. Set the main file path to `app.py`
5. Click "Deploy!"

### Alternative: Deploy to Render

1. Push your code to a GitHub repository
2. Go to [render.com](https://render.com) and sign up/log in
3. Click "New+" and select "Web Service"
4. Connect your GitHub repository
5. Set the following options:
   - Name: Your app name
   - Runtime: Python 3
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
6. Click "Create Web Service"

## Using Larger Datasets (Optional)

To use a larger dataset:

1. Replace `sample_data.csv` with your dataset
2. Ensure your CSV has these columns: `id`, `title`, `description`, `tags`, `category`
3. Optional: Add an `image_url` column for images
4. Restart the app

Compatible datasets include IMDb movie listings, Goodreads book data, or e-commerce product catalogs formatted appropriately.

## Project Structure

```
recommendation/
├── app.py                 # Streamlit web application
├── recommender.py         # Recommendation engine logic
├── utils.py               # Helper functions
├── test_recommender.py    # Unit tests
├── requirements.txt       # Python dependencies
├── Procfile               # Deployment configuration
├── README.md              # This file
└── sample_data.csv        # Sample dataset
```

## Running Tests

To run the unit tests:

```
python -m unittest test_recommender.py
```

## API Reference

### Main Functions

- `recommend_by_text(preferences: str, top_k: int = 5)` - Get recommendations based on text preferences
- `recommender.get_similar_items(item_id: int, top_k: int = 5)` - Find items similar to a given item
- `recommender.get_categories()` - Get all available categories
- `recommender.get_items_by_category(category: str, top_k: int = 10)` - Get items from a specific category

## How It Works

1. **Data Preparation**: The system loads item data and combines text features (title, description, tags, category)
2. **Text Preprocessing**: Text is cleaned, tokenized, stemmed, and stopwords are removed using NLTK
3. **Feature Extraction**: TF-IDF vectorization converts text to numerical features
4. **Similarity Calculation**: Cosine similarity measures the similarity between user preferences and items
5. **Recommendation Generation**: Top-K items with highest similarity scores are returned with explanations

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check that your dataset has the required columns
3. Verify Python 3.8+ is installed
4. For deployment issues, check the logs for specific error messages