import streamlit as st  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import io
import base64
import requests
from io import BytesIO
import cv2

def init_page():
    st.set_page_config(page_title="CHRISPO '25 Analysis", layout="wide")
  
@st.cache_data
def load_data():
    feedback_texts = [
        "Great organization", "Need better facilities", "Excellent competition",
        "Well managed", "Could improve timing", "Amazing experience",
        "Good atmosphere", "Need better refreshments", "Outstanding event",
        "Professional management",
    ] * 10
    
    return pd.DataFrame({
        'Sport': ['Cricket', 'Football', 'Basketball', 'Volleyball', 'Athletics'] * 20,
        'College': ['College A', 'College B', 'College C', 'College D', 'College E'] * 20,
        'State': ['Kerala', 'Tamil Nadu', 'Karnataka', 'Maharashtra', 'Delhi'] * 20,
        'Day': ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'] * 20,
        'Participants': np.random.randint(10, 100, 100),
        'Feedback': feedback_texts
    })

def create_filters(data):
    with st.sidebar:
        st.header("Filters")
        selected_sports = st.multiselect("Select Sports", data['Sport'].unique())
        selected_states = st.multiselect("Select States", data['State'].unique())
        selected_colleges = st.multiselect("Select Colleges", data['College'].unique())
    return selected_sports, selected_states, selected_colleges

def filter_data(data, sports, states, colleges):
    mask = pd.Series(True, index=data.index)
    if sports: mask &= data['Sport'].isin(sports)
    if states: mask &= data['State'].isin(states)
    if colleges: mask &= data['College'].isin(colleges)
    return data[mask]

def create_participation_charts(filtered_data):
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(filtered_data.groupby('Sport')['Participants'].sum().reset_index(),
                      x='Sport', y='Participants', title='Sports-wise Participation')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.pie(filtered_data.groupby('State')['Participants'].sum().reset_index(),
                      values='Participants', names='State', title='State-wise Distribution')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.bar(filtered_data.groupby('College')['Participants'].sum().reset_index(),
                      x='College', y='Participants', title='College-wise Participation')
        st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = px.line(filtered_data.groupby('Day')['Participants'].mean().reset_index(),
                       x='Day', y='Participants', title='Daily Participation Trend')
        st.plotly_chart(fig4, use_container_width=True)

def create_heatmap(filtered_data):
    fig5 = px.density_heatmap(filtered_data, x='Sport', y='College', z='Participants',
                             title='Sport vs College Participation Heat Map')
    st.plotly_chart(fig5, use_container_width=True)

def display_metrics(filtered_data):
    st.header("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Participants", filtered_data['Participants'].sum())
    with col2:
        st.metric("Total Sports", filtered_data['Sport'].nunique())
    with col3:
        st.metric("Total Colleges", filtered_data['College'].nunique())

def create_word_cloud(filtered_data):
    st.header("Feedback Analysis")
    selected_sport_feedback = st.selectbox("Select Sport for Feedback Analysis", 
                                         filtered_data['Sport'].unique())
    sport_feedback = filtered_data[filtered_data['Sport'] == selected_sport_feedback]['Feedback']
    if len(sport_feedback) > 0:
        # Create word frequency dictionary
        text = ' '.join(sport_feedback)
        words = text.split()
        word_freq = {}
        for word in words:
            word = word.lower()
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
        
        # Convert to dataframe for plotly
        word_df = pd.DataFrame(list(word_freq.items()), columns=['word', 'count'])
        word_df = word_df.sort_values('count', ascending=False).head(50)
        
        # Create scatter plot with size based on word frequency
        fig = px.scatter(word_df, 
                        x=np.random.rand(len(word_df)),  # Random x positions
                        y=np.random.rand(len(word_df)),  # Random y positions
                        size='count',
                        text='word',
                        size_max=60,
                        title=f'Word Cloud for {selected_sport_feedback}')
        
        fig.update_traces(textposition='top center')
        fig.update_layout(
            showlegend=False,
            xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            yaxis={'showgrid': False, 'zeroline': False, 'visible': False}
        )
        st.plotly_chart(fig, use_container_width=True)

def analyze_sentiment(filtered_data):
    st.subheader("Feedback Sentiment by Sport")
    try:
        feedback_sentiment = filtered_data.copy()
        feedback_sentiment['Sentiment'] = feedback_sentiment['Feedback'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        fig_sentiment = px.box(feedback_sentiment, x='Sport', y='Sentiment',
                             title='Sentiment Distribution by Sport')
        st.plotly_chart(fig_sentiment, use_container_width=True)
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")

def analyze_common_words(filtered_data):
    st.subheader("Most Common Words by Sport")
    sport_words = {}
    for sport in filtered_data['Sport'].unique():
        sport_text = ' '.join(filtered_data[filtered_data['Sport'] == sport]['Feedback'])
        word_freq = pd.Series(sport_text.lower().split()).value_counts().head(5)
        sport_words[sport] = word_freq

    cols = st.columns(len(sport_words))
    for i, (sport, words) in enumerate(sport_words.items()):
        with cols[i]:
            st.write(f"**{sport}**")
            st.write(words)

def create_landing_page():
    st.markdown("""
    <style>
    .hero {
        padding: 2rem;
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #f0f2f6;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.8;
    }
    </style>
    
    <div class="hero">
        <div class="hero-title">CHRISPO '25</div>
        <div class="hero-subtitle">Inter-College Tournament Analysis Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    # About section
    st.markdown("""
    <div class="section-header">
        <h2>About CHRISPO '25</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        Welcome to the official analytics dashboard for CHRISPO '25, the premier inter-college 
        tournament bringing together athletic talent from across the nation. This platform 
        provides comprehensive insights into participation trends, feedback analysis, and 
        tournament statistics.
        
        Key Features:
        - Real-time participation analytics
        - Interactive data visualization
        - Sentiment analysis of feedback
        - College-wise performance metrics
        """)
    with col2:
        st.info("📊 Analyze participation trends\n\n"
                "💭 Process participant feedback\n\n"
                "📈 Track engagement metrics")

def process_image(image, brightness=1.0, contrast=1.0, sharpness=1.0, filter_type=None):
    img = Image.open(image)
    
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    
    if filter_type == "Blur":
        img = img.filter(ImageFilter.BLUR)
    elif filter_type == "Sharpen":
        img = img.filter(ImageFilter.SHARPEN)
    elif filter_type == "Edge Enhance":
        img = img.filter(ImageFilter.EDGE_ENHANCE)
    
    return img

def process_image_advanced(image, operations=None):
    """Advanced image processing with OpenCV and PIL"""
    img = Image.open(image)
    img_array = np.array(img)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    processed = None
    if operations:
        for op, params in operations.items():
            if op == 'edge_detection':
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                processed = cv2.Canny(gray, params['threshold1'], params['threshold2'])
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            elif op == 'blur':
                processed = cv2.GaussianBlur(img_array, (params['kernel'], params['kernel']), 0)
            elif op == 'cartoon':
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                           cv2.THRESH_BINARY, 9, 9)
                color = cv2.bilateralFilter(img_array, params['d'], 300, 300)
                processed = cv2.bitwise_and(color, color, mask=edges)
            elif op == 'rotate':
                processed = Image.fromarray(img_array).rotate(params['angle'])
                processed = np.array(processed)
            elif op == 'watermark':
                processed = Image.fromarray(img_array)
                draw = ImageDraw.Draw(processed)
                draw.text((10, 10), params['text'], fill='white')
                processed = np.array(processed)
    
    return Image.fromarray(processed if processed is not None else img_array)

def create_image_gallery():
    st.markdown("""
    <div class="section-header">
        <h2>Sports Image Gallery</h2>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload Day-wise Images", 
                                    type=['png', 'jpg', 'jpeg'], 
                                    accept_multiple_files=True)
    
    if uploaded_files:
        for i, file in enumerate(uploaded_files):
            col1, col2 = st.columns(2)
            with col1:
                st.image(file, caption=f"Day {i+1} Image", use_column_width=True)
            with col2:
                st.subheader("Image Processing")
                brightness = st.slider(f"Brightness {i+1}", 0.0, 2.0, 1.0)
                contrast = st.slider(f"Contrast {i+1}", 0.0, 2.0, 1.0)
                sharpness = st.slider(f"Sharpness {i+1}", 0.0, 2.0, 1.0)
                filter_type = st.selectbox(f"Filter {i+1}", 
                                         [None, "Blur", "Sharpen", "Edge Enhance"])
                
                processed_img = process_image(file, brightness, contrast, 
                                           sharpness, filter_type)
                st.image(processed_img, caption="Processed Image", 
                        use_column_width=True)
                
                buf = io.BytesIO()
                processed_img.save(buf, format="PNG")
                btn = st.download_button(
                    label="Download Processed Image",
                    data=buf.getvalue(),
                    file_name=f"processed_image_{i+1}.png",
                    mime="image/png"
                )

def create_image_analysis():
    st.markdown("""
    <div class="section-header">
        <h2>Advanced Image Analysis</h2>
    </div>
    
    <style>
    .image-tools {
        background: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .tool-section {
        margin: 1rem 0;
        padding: 1rem;
        border: 1px solid #eee;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an image for analysis", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Original Image", use_column_width=True)
            
        with col2:
            analysis_type = st.selectbox("Select Analysis Type", 
                                       ["Basic Enhancement", "Edge Detection", 
                                        "Artistic Effects", "Advanced Operations"])
            
            operations = {}
            
            if analysis_type == "Basic Enhancement":
                brightness = st.slider("Brightness", 0.0, 2.0, 1.0)
                contrast = st.slider("Contrast", 0.0, 2.0, 1.0)
                sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0)
                operations = {
                    'enhance': {
                        'brightness': brightness,
                        'contrast': contrast,
                        'sharpness': sharpness
                    }
                }
            
            elif analysis_type == "Edge Detection":
                threshold1 = st.slider("Edge Threshold 1", 0, 255, 100)
                threshold2 = st.slider("Edge Threshold 2", 0, 255, 200)
                operations = {
                    'edge_detection': {
                        'threshold1': threshold1,
                        'threshold2': threshold2
                    }
                }
            
            elif analysis_type == "Artistic Effects":
                effect = st.selectbox("Choose Effect", 
                                    ["Cartoon", "Watermark", "Sketch"])
                if effect == "Cartoon":
                    operations = {
                        'cartoon': {'d': st.slider("Detail", 5, 15, 9)}
                    }
                elif effect == "Watermark":
                    operations = {
                        'watermark': {'text': st.text_input("Watermark Text", "CHRISPO '25")}
                    }
            
            elif analysis_type == "Advanced Operations":
                operation = st.selectbox("Choose Operation", 
                                       ["Rotate", "Color Analysis", "Histogram"])
                if operation == "Rotate":
                    operations = {
                        'rotate': {'angle': st.slider("Rotation Angle", -180, 180, 0)}
                    }
                elif operation == "Color Analysis":
                    st.write("Color Distribution")
                    img = Image.open(uploaded_file)
                    img_array = np.array(img)
                    fig = px.histogram(img_array.reshape(-1, 3), 
                                     nbins=50,
                                     labels={'value': 'Pixel Value', 
                                            'count': 'Frequency'})
                    st.plotly_chart(fig)
            
            processed_img = process_image_advanced(uploaded_file, operations)
            st.image(processed_img, caption="Processed Image", use_column_width=True)

def fetch_sports_images(sport, count=6):
    """Fetch sports images using sport categories"""
    sports_images = {
        'Cricket': [
            ('https://source.unsplash.com/800x600/?cricket-match', 'Cricket Match'),
            ('https://source.unsplash.com/800x600/?cricket-stadium', 'Cricket Stadium'),
            ('https://source.unsplash.com/800x600/?cricket-bat', 'Cricket Equipment'),
        ],
        'Football': [
            ('https://source.unsplash.com/800x600/?football-match', 'Football Match'),
            ('https://source.unsplash.com/800x600/?soccer-stadium', 'Football Stadium'),
            ('https://source.unsplash.com/800x600/?soccer-ball', 'Football Game'),
        ],
        'Basketball': [
            ('https://source.unsplash.com/800x600/?basketball-game', 'Basketball Game'),
            ('https://source.unsplash.com/800x600/?basketball-court', 'Basketball Court'),
            ('https://source.unsplash.com/800x600/?basketball', 'Basketball'),
        ],
        'Volleyball': [
            ('https://source.unsplash.com/800x600/?volleyball-game', 'Volleyball Match'),
            ('https://source.unsplash.com/800x600/?volleyball-court', 'Volleyball Court'),
            ('https://source.unsplash.com/800x600/?volleyball', 'Volleyball'),
        ],
        'Athletics': [
            ('https://source.unsplash.com/800x600/?running-track', 'Athletics Track'),
            ('https://source.unsplash.com/800x600/?athletics', 'Athletics Event'),
            ('https://source.unsplash.com/800x600/?sprint', 'Sprint'),
        ]
    }
    return sports_images.get(sport, [])[:count]

def create_sports_gallery():
    st.markdown("""
    <div class="section-header">
        <h2>CHRISPO '25 Sports Gallery</h2>
    </div>
    
    <style>
    .gallery-container {
        padding: 2rem 0;
    }
    .sport-section {
        margin-bottom: 3rem;
    }
    .sport-title {
        color: #1e3c72;
        font-size: 1.5rem;
        margin: 1rem 0;
        padding: 0.5rem 1rem;
        background: #f8f9fa;
        border-radius: 5px;
        border-left: 4px solid #1e3c72;
    }
    .gallery-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
    }
    .gallery-item {
        position: relative;
        overflow: hidden;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        aspect-ratio: 16/9;
    }
    .gallery-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    .gallery-item img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .image-caption {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 0.75rem;
        background: linear-gradient(transparent, rgba(0,0,0,0.8));
        color: white;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    sports = ["Cricket", "Football", "Basketball", "Volleyball", "Athletics"]
    
    for sport in sports:
        images = fetch_sports_images(sport)
        if images:
            st.markdown(f"<div class='sport-section'><h3 class='sport-title'>{sport}</h3>", unsafe_allow_html=True)
            st.markdown("<div class='gallery-grid'>", unsafe_allow_html=True)
            
            for img_url, caption in images:
                st.markdown(f"""
                <div class="gallery-item">
                    <img src="{img_url}" alt="{caption}">
                    <div class="image-caption">{caption}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("</div></div>", unsafe_allow_html=True)

def create_navigation():
    st.sidebar.markdown("""
    <style>
    .sidebar-nav {
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .nav-item {
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    return st.sidebar.radio(
        "Navigation",
        ["Home", "Participation Analysis", "Feedback Analysis", 
         "Image Analysis", "Sports Gallery", "About"],
        format_func=lambda x: f"📊 {x}" if x == "Participation Analysis"
                    else f"💭 {x}" if x == "Feedback Analysis"
                    else f"🖼️ {x}" if x == "Image Analysis"
                    else f"🏆 {x}" if x == "Sports Gallery"
                    else f"ℹ️ {x}" if x == "About"
                    else f"🏠 {x}"
    )

def main():
    init_page()
    create_landing_page()
    
    page = create_navigation()
    
    data = load_data()
    selected_sports, selected_states, selected_colleges = create_filters(data)
    filtered_data = filter_data(data, selected_sports, selected_states, selected_colleges)
    
    if page == "Home":
        st.markdown("""
        <div class="section-header">
            <h2>Tournament Overview</h2>
        </div>
        """, unsafe_allow_html=True)
        display_metrics(filtered_data)
        
    elif page == "Participation Analysis":
        st.markdown("""
        <div class="section-header">
            <h2>Participation Analytics</h2>
        </div>
        """, unsafe_allow_html=True)
        create_participation_charts(filtered_data)
        create_heatmap(filtered_data)
        
    elif page == "Feedback Analysis":
        st.markdown("""
        <div class="section-header">
            <h2>Feedback Insights</h2>
        </div>
        """, unsafe_allow_html=True)
        create_word_cloud(filtered_data)
        analyze_sentiment(filtered_data)
        analyze_common_words(filtered_data)
        
    elif page == "Image Analysis":
        create_image_analysis()
        
    elif page == "Sports Gallery":
        create_sports_gallery()
        
    else:
        st.markdown("""
        <div class="section-header">
            <h2>About the Project</h2>
        </div>
        """, unsafe_allow_html=True)
        st.write("""
        This dashboard is developed by the Website Development team for CHRISPO '25. 
        It serves as a comprehensive tool for analyzing tournament data and providing 
        actionable insights to the core team.
        
        Technologies Used:
        - Streamlit for web application development
        - Python for data analysis
        - Plotly for interactive visualizations
        - Natural Language Processing for feedback analysis
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()
