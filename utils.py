import pandas as pd
import streamlit as st


def load_sample_data(filepath: str = "sample_data.csv") -> pd.DataFrame:
    """
    Load sample data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {filepath}")


def get_sample_prompts() -> list:
    """
    Get sample prompts for quick selection in the UI.
    
    Returns:
        list: List of sample prompt strings
    """
    return [
        "movies with action and adventure",
        "dramas about friendship and hope",
        "sci-fi films with philosophical themes",
        "crime movies with complex characters",
        "fantasy adventures with epic battles"
    ]


def validate_data_format(df: pd.DataFrame) -> bool:
    """
    Validate that the data has the required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_columns = ['id', 'title', 'description', 'tags', 'category']
    return all(col in df.columns for col in required_columns)


def display_item_card(item: dict, show_score: bool = True):
    """
    Display an item as a card in the Streamlit UI.
    
    Args:
        item (dict): Item data to display
        show_score (bool): Whether to show the similarity score
    """
    with st.container():
        st.markdown("---")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display image if available
            if item.get('image_url'):
                st.image(item['image_url'], width=150)
            else:
                # Placeholder if no image
                st.image("https://placehold.co/150x200?text=No+Image", width=150)
            if show_score and 'score' in item:
                st.metric("Match Score", f"{item['score']:.3f}")
        
        with col2:
            # Display item details
            st.subheader(item['title'])
            st.write(f"**Category:** {item['category']}")
            st.write(f"**Description:** {item['description']}")
            if 'explanation' in item:
                st.info(f"**Why Recommended:** {item['explanation']}")


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame to summarize
        
    Returns:
        dict: Summary statistics
    """
    return {
        'total_items': len(df),
        'categories': df['category'].nunique(),
        'category_list': df['category'].unique().tolist(),
        'avg_description_length': df['description'].str.len().mean()
    }