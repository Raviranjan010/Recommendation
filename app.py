import streamlit as st
# import pandas as pd
from recommender import recommend_by_text, recommender
from utils import get_sample_prompts, display_item_card, get_data_summary
import time


# Configure the page
st.set_page_config(
    page_title="Content Recommendation System",
    page_icon="🎬",
    layout="wide"
)


def main():
    """
    Main function to run the Streamlit app.
    """
    # App title and description
    st.title("🎬 Content-Based Recommendation System")
    st.markdown("""
    Discover content tailored to your interests! Enter your preferences below 
    to get personalized recommendations.
    """)
    
    # Load data summary
    try:
        data_summary = get_data_summary(recommender.df)
        st.markdown(f"*Currently serving **{data_summary['total_items']}** items across **{data_summary['categories']}** categories*")
    except:
        st.markdown("*Loading dataset...*")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Recommendations", "Browse by Category", "Similar Items"])
    
    with tab1:
        # Sidebar for inputs
        with st.sidebar:
            st.header("User Preferences")
            
            # Sample prompts for quick selection
            st.subheader("Quick Picks")
            sample_prompts = get_sample_prompts()
            
            # Create buttons for sample prompts
            selected_prompt = None
            for i, prompt in enumerate(sample_prompts):
                if st.button(prompt, key=f"prompt_{i}"):
                    selected_prompt = prompt
                    
            # Text input for custom preferences
            st.subheader("Custom Preference")
            user_input = st.text_input(
                "Describe what you're interested in:",
                value=selected_prompt if selected_prompt else "",
                placeholder="e.g., movies with action and adventure...",
                key="user_preference"
            )
            
            # Number of recommendations
            top_k = st.slider("Number of recommendations", 1, 20, 5)
            
            # Advanced options
            with st.expander("Advanced Options"):
                use_stemming = st.checkbox("Use advanced text processing", value=True)
                min_similarity = st.slider("Minimum similarity threshold", 0.0, 1.0, 0.0, 0.05)
            
            # Run recommendation button
            run_button = st.button("Get Recommendations", type="primary")
        
        # Main content area
        if run_button and user_input:
            with st.spinner("Finding the best recommendations for you..."):
                # Simulate some processing time for better UX
                time.sleep(0.5)
                
                # Get recommendations
                try:
                    recommendations = recommend_by_text(user_input, top_k)
                    
                    # Filter by minimum similarity if set
                    if min_similarity > 0:
                        recommendations = [rec for rec in recommendations if rec['score'] >= min_similarity]
                    
                    # Display results
                    st.header(f"Top {len(recommendations)} Recommendations")
                    
                    if recommendations:
                        for i, rec in enumerate(recommendations):
                            display_item_card(rec)
                        
                        # Export to CSV option
                        st.markdown("---")
                        st.subheader("Export Results")
                        
                        # Convert to DataFrame for export
                        df_export = pd.DataFrame(recommendations)
                        
                        # Create CSV download button
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label="Download Recommendations as CSV",
                            data=csv,
                            file_name="recommendations.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No recommendations found matching your criteria. Try adjusting your preferences or lowering the similarity threshold.")
                        
                except Exception as e:
                    st.error(f"An error occurred while generating recommendations: {str(e)}")
                    
        elif run_button and not user_input:
            st.warning("Please enter your preferences to get recommendations.")
        else:
            # Show sample data preview
            st.info("Enter your preferences in the sidebar and click 'Get Recommendations' to start!")
            if recommender.df is not None:
                st.subheader("Sample Items from Our Collection")
                sample_items = recommender.df.sample(min(3, len(recommender.df))).to_dict('records')
                for item in sample_items:
                    # Add placeholder values for missing keys
                    item.setdefault('score', 0.0)
                    item.setdefault('explanation', 'Sample item from collection')
                    display_item_card(item, show_score=False)
            
    with tab2:
        st.header("Browse by Category")
        try:
            categories = recommender.get_categories()
            selected_category = st.selectbox("Select a category:", categories)
            
            if selected_category:
                items = recommender.get_items_by_category(selected_category, 10)
                st.subheader(f"Items in {selected_category}")
                
                if items:
                    for item in items:
                        display_item_card(item, show_score=False)
                else:
                    st.info("No items found in this category.")
        except Exception as e:
            st.error(f"Error loading categories: {str(e)}")
    
    with tab3:
        st.header("Find Similar Items")
        st.markdown("Select an item to find similar items in our collection.")
        
        try:
            # Create a selectbox with all items
            if recommender.df is not None:
                item_options = {f"{row['title']} ({row['category']})": row['id'] 
                               for _, row in recommender.df.iterrows()}
                selected_item_name = st.selectbox("Select an item:", list(item_options.keys()))
                
                if selected_item_name:
                    selected_item_id = item_options[selected_item_name]
                    similar_items = recommender.get_similar_items(selected_item_id, 5)
                    
                    if similar_items:
                        st.subheader("Similar Items")
                        for item in similar_items:
                            display_item_card(item)
                    else:
                        st.info("No similar items found.")
        except Exception as e:
            st.error(f"Error finding similar items: {str(e)}")
        
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How it works:**
    - This system uses content-based filtering
    - It analyzes text features (title, description, tags)
    - Recommendations are based on similarity to your preferences
    """)


if __name__ == "__main__":
    main()