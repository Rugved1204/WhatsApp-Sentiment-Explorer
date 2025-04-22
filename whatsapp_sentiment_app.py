import streamlit as st
import pandas as pd
import re
import emoji
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime

# Download NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

# 1. Load WhatsApp chat file
st.title("📱 WhatsApp Chat Sentiment Analyzer")
st.markdown("Upload your WhatsApp chat export to analyze sentiment patterns and visualize chat statistics.")

uploaded_file = st.file_uploader("Upload your WhatsApp chat file (.txt)", type=["txt"])

# 2. Text cleaning
def clean_text(text):
    text = emoji.demojize(text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

# 3. Parse the WhatsApp chat with enhanced metadata
def parse_chat(file):
    messages = []
    dates = []
    times = []
    senders = []
    
    date_pattern = r'(\d{1,2}/\d{1,2}/\d{2,4})'
    time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)'
    
    for line in file:
        date_match = re.search(date_pattern, line)
        time_match = re.search(time_pattern, line)
        
        parts = re.split(r'\s-\s', line, maxsplit=1)
        if len(parts) > 1:
            date_time = parts[0] if date_match else "Unknown"
            msg_part = parts[1]
            
            if ': ' in msg_part:
                sender, message = msg_part.split(': ', 1)
                messages.append(message)
                senders.append(sender)
                
                if date_match and time_match:
                    dates.append(date_match.group(1))
                    times.append(time_match.group(1))
                else:
                    dates.append("Unknown")
                    times.append("Unknown")
    
    return pd.DataFrame({
        'date': dates,
        'time': times,
        'sender': senders,
        'message': messages
    })

# 4. Enhanced Wordcloud generator with options
def show_wordcloud(messages, title='Word Cloud', colormap='viridis', max_words=200):
    text = ' '.join(messages)
    
    # Remove common stop words
    stop_words = ['the', 'and', 'is', 'in', 'it', 'to', 'that', 'was', 'for', 'on', 'with', 'at', 'this', 'but']
    
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap=colormap,
        max_words=max_words,
        stopwords=stop_words
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

# 5. Load or train model with improved visualization
@st.cache_resource
def train_model(df):
    # Use VADER for initial sentiment labeling
    sia = SentimentIntensityAnalyzer()
    
    # Get sentiment scores and convert to labels
    df['sentiment_score'] = df['message'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['label'] = df['sentiment_score'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))
    
    # Prepare data for training
    df['text_clean'] = df['message'].apply(clean_text)
    
    X_train, X_test, y_train, y_test = train_test_split(df['text_clean'], df['label'], test_size=0.2, random_state=42)
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Save model artifacts
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    
    # Model evaluation
    X_test_tfidf = tfidf.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, tfidf, report, df

# 6. Time-based sentiment analysis
def plot_sentiment_over_time(df):
    if 'date' in df.columns and df['date'].iloc[0] != "Unknown":
        # Convert date strings to datetime objects
        try:
            # Try different date formats
            date_formats = ['%d/%m/%Y', '%m/%d/%Y', '%d/%m/%y', '%m/%d/%y']
            for fmt in date_formats:
                try:
                    df['date_parsed'] = pd.to_datetime(df['date'], format=fmt)
                    break
                except:
                    continue
                
            # Group by date and calculate average sentiment
            daily_sentiment = df.groupby(df['date_parsed'].dt.date)['sentiment_score'].mean().reset_index()
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(daily_sentiment['date_parsed'], daily_sentiment['sentiment_score'], marker='o', linestyle='-')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            plt.fill_between(
                daily_sentiment['date_parsed'],
                daily_sentiment['sentiment_score'],
                0,
                where=(daily_sentiment['sentiment_score'] >= 0),
                interpolate=True,
                color='green',
                alpha=0.3
            )
            plt.fill_between(
                daily_sentiment['date_parsed'],
                daily_sentiment['sentiment_score'],
                0,
                where=(daily_sentiment['sentiment_score'] <= 0),
                interpolate=True,
                color='red',
                alpha=0.3
            )
            plt.title('Sentiment Trend Over Time')
            plt.xlabel('Date')
            plt.ylabel('Average Sentiment Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error processing dates: {e}")
            st.info("Skipping time-based analysis due to date format issues")
    else:
        st.info("Date information not available in the chat. Skipping time-based analysis.")

# 7. Sentiment distribution by sender
def plot_sentiment_by_sender(df):
    if len(df['sender'].unique()) > 1:
        # Calculate average sentiment for each sender
        sender_sentiment = df.groupby('sender')['sentiment_score'].agg(['mean', 'count']).reset_index()
        sender_sentiment = sender_sentiment.sort_values(by='mean')
        
        # Only display top senders for readability
        top_senders = sender_sentiment.nlargest(10, 'count')
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.barh(top_senders['sender'], top_senders['mean'], color=plt.cm.RdYlGn(
            (top_senders['mean'] + 1) / 2))
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Average Sentiment Score by Sender')
        plt.xlabel('Sentiment Score (-1: Very Negative, +1: Very Positive)')
        plt.tight_layout()
        st.pyplot(plt)
        
        # Add message count information
        st.markdown("### Message Count by Sender")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='count', y='sender', data=top_senders)
        plt.title('Number of Messages per Sender')
        plt.xlabel('Message Count')
        plt.tight_layout()
        st.pyplot(fig)

# 8. Confusion matrix visualization
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'] if 'neutral' in set(y_test) else ['Negative', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'] if 'neutral' in set(y_test) else ['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# 9. Message length analysis
def analyze_message_length(df):
    df['msg_length'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot message length histogram
    sns.histplot(data=df, x='msg_length', kde=True, ax=ax1)
    ax1.set_title('Distribution of Message Lengths')
    ax1.set_xlabel('Characters per Message')
    
    # Plot word count histogram
    sns.histplot(data=df, x='word_count', kde=True, ax=ax2)
    ax2.set_title('Distribution of Word Counts')
    ax2.set_xlabel('Words per Message')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Relationship between message length and sentiment
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='word_count', y='sentiment_score', hue='label', 
                    palette={'positive': 'green', 'negative': 'red', 'neutral': 'blue'}, alpha=0.6)
    plt.title('Relationship Between Message Length and Sentiment')
    plt.xlabel('Word Count')
    plt.ylabel('Sentiment Score')
    st.pyplot(fig)

# 10. Top keywords by sentiment
def plot_top_keywords(df, tfidf):
    # Get top words for each sentiment category
    def get_top_words(category, n=10):
        texts = df[df['label'] == category]['text_clean']
        text = ' '.join(texts)
        words = text.split()
        word_count = Counter(words)
        return word_count.most_common(n)
    
    if 'neutral' in set(df['label']):
        categories = ['positive', 'neutral', 'negative']
    else:
        categories = ['positive', 'negative']
    
    # Create a figure for word frequency
    fig, axes = plt.subplots(len(categories), 1, figsize=(10, 4*len(categories)))
    
    if len(categories) == 1:
        axes = [axes]
    
    for i, category in enumerate(categories):
        top_words = get_top_words(category)
        if top_words:  # Only proceed if we have words for this category
            words, counts = zip(*top_words)
            sns.barplot(x=list(counts), y=list(words), ax=axes[i])
            axes[i].set_title(f'Top Words in {category.capitalize()} Messages')
            axes[i].set_xlabel('Count')
    
    plt.tight_layout()
    st.pyplot(fig)

# Main app flow
if uploaded_file:
    raw = uploaded_file.read().decode('utf-8').split('\n')
    
    with st.spinner("Processing chat data..."):
        # Parse chat into dataframe
        chat_df = parse_chat(raw)
        
        # Display basic chat statistics
        st.subheader("📊 Chat Overview")
        st.write(f"Total messages: {len(chat_df)}")
        
        if len(chat_df['sender'].unique()) > 1:
            st.write(f"Number of participants: {len(chat_df['sender'].unique())}")
            
        with st.expander("Preview of chat data"):
            st.dataframe(chat_df.head())
        
        # Train the sentiment model
        model, vectorizer, report, processed_df = train_model(chat_df)
        
        # Display visualizations in tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "WordCloud", "Sentiment Distribution", "Sentiment Over Time", 
            "Sender Analysis", "Message Analysis", "Model Performance"
        ])
        
        with tab1:
            st.subheader("📝 Word Cloud Visualization")
            colormap_option = st.selectbox(
                "Select color scheme:",
                options=["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Greens", "Reds"]
            )
            
            # Create separate wordclouds for each sentiment
            st.subheader("Complete Chat Word Cloud")
            show_wordcloud(processed_df['message'], colormap=colormap_option)
            
            # Only show sentiment-specific wordclouds if we have enough data
            col1, col2 = st.columns(2)
            with col1:
                positive_msgs = processed_df[processed_df['label'] == 'positive']['message']
                if len(positive_msgs) > 5:
                    st.subheader("Positive Messages")
                    show_wordcloud(positive_msgs, title="Positive Sentiment Words", colormap="Greens")
            
            with col2:
                negative_msgs = processed_df[processed_df['label'] == 'negative']['message']
                if len(negative_msgs) > 5:
                    st.subheader("Negative Messages")
                    show_wordcloud(negative_msgs, title="Negative Sentiment Words", colormap="Reds")
        
        with tab2:
            st.subheader("😊 Sentiment Distribution")
            
            # Pie chart of sentiment distribution
            fig, ax = plt.subplots(figsize=(8, 8))
            sentiment_counts = processed_df['label'].value_counts()
            colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
            used_colors = [colors[label] for label in sentiment_counts.index]
            
            wedges, texts, autotexts = ax.pie(
                sentiment_counts, 
                labels=sentiment_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=used_colors
            )
            ax.axis('equal')
            plt.setp(autotexts, size=10, weight='bold')
            plt.title('Sentiment Distribution in Chat')
            st.pyplot(fig)
            
            # Histogram of sentiment scores
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=processed_df, x='sentiment_score', kde=True, color='teal')
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title('Distribution of Sentiment Scores')
            plt.xlabel('Sentiment Score (-1: Very Negative, +1: Very Positive)')
            st.pyplot(fig)
            
            # Show top keywords by sentiment
            st.subheader("Top Keywords by Sentiment")
            plot_top_keywords(processed_df, vectorizer)
        
        with tab3:
            st.subheader("📅 Sentiment Trends Over Time")
            plot_sentiment_over_time(processed_df)
        
        with tab4:
            st.subheader("👥 Sender Analysis")
            plot_sentiment_by_sender(processed_df)
        
        with tab5:
            st.subheader("📏 Message Length Analysis")
            analyze_message_length(processed_df)
        
        with tab6:
            st.subheader("🎯 Model Performance")
            
            # Classification report as a heatmap
            report_df = pd.DataFrame(report).T
            if 'support' in report_df.columns:
                report_df = report_df.drop('support', axis=1)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(report_df.iloc[:-3], annot=True, cmap='Blues', fmt='.2f')
            plt.title('Classification Report')
            st.pyplot(fig)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            X_test_vec = vectorizer.transform(processed_df['text_clean'])
            y_true = processed_df['label']
            y_pred = model.predict(X_test_vec)
            plot_confusion_matrix(y_true, y_pred)
    
    # Interactive message sentiment analysis
    st.subheader("🔍 Try it yourself!")
    user_input = st.text_area("Enter a message to analyze:", height=100)
    
    if user_input:
        clean = clean_text(user_input)
        
        # VADER analysis for score
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(user_input)
        
        # Model prediction
        vec = vectorizer.transform([clean])
        prediction = model.predict(vec)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Predicted Sentiment", 
                value=prediction.upper(),
                delta=f"{sentiment_score['compound']:.2f}"
            )
            
        with col2:
            # Create a gauge chart for sentiment
            fig, ax = plt.subplots(figsize=(4, 0.7))
            ax.barh(0, 2, left=-1, height=0.5, color='lightgray')
            ax.barh(0, sentiment_score['compound'] + 1, left=-1, height=0.5, 
                   color=plt.cm.RdYlGn((sentiment_score['compound'] + 1) / 2))
            ax.set_xlim(-1, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xticklabels(['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
            ax.set_yticks([])
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            st.pyplot(fig)
        
        # Show detailed sentiment scores
        st.json({
            'positive': sentiment_score['pos'],
            'neutral': sentiment_score['neu'],
            'negative': sentiment_score['neg'],
            'compound': sentiment_score['compound']
        })
else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a WhatsApp chat export file to begin analysis.")
    st.markdown("""
    ### How to export your WhatsApp chat:
    1. Open the chat you want to analyze
    2. Tap the three dots (⋮) > More > Export chat
    3. Choose "Without Media"
    4. Save the file and upload it here
    
    ### Features:
    - Sentiment analysis of messages
    - WordCloud visualization
    - Sentiment trends over time
    - Sender-based analysis
    - Message length statistics
    """)
