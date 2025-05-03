import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from datetime import datetime, timedelta
import random  # For simulated data

# Streamlit page configuration
st.set_page_config(page_title="Startup Brand Sentiment Analyzer", layout="wide")

# Initialize sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Simulated social media scraping function (replace with tweepy/BeautifulSoup for real data)
def scrape_social_media(brand, platforms, days=7):
    # Simulated data for X and Yelp mentions
    dates = [datetime.now() - timedelta(days=x) for x in range(days)]
    data = []
    for date in dates:
        for platform in platforms:
            num_mentions = random.randint(5, 20)
            for _ in range(num_mentions):
                text = f"{brand} is {random.choice(['great', 'terrible', 'okay', 'awesome', 'bad'])} on {platform}!"
                data.append({
                    "date": date,
                    "platform": platform,
                    "text": text
                })
    return pd.DataFrame(data)

# Analyze sentiment of mentions
def analyze_sentiment(df):
    sentiments = []
    scores = []
    for text in df["text"]:
        result = sentiment_analyzer(text)[0]
        label = result["label"]
        score = result["score"]
        sentiments.append(label)
        scores.append(score if label == "POSITIVE" else -score)
    df["sentiment"] = sentiments
    df["sentiment_score"] = scores
    return df

# Generate sentiment trend plot
def plot_sentiment_trends(df, group_by):
    df["date"] = df["date"].dt.date
    trend_data = df.groupby([group_by, "date"])["sentiment_score"].mean().reset_index()
    fig = px.line(trend_data, x="date", y="sentiment_score", color=group_by,
                  title=f"Sentiment Trends by {group_by}",
                  labels={"sentiment_score": "Average Sentiment Score", "date": "Date"})
    fig.update_layout(yaxis_range=[-1, 1])
    return fig

# Check for negative sentiment spikes
def detect_negative_spikes(df, threshold=-0.5):
    daily_sentiment = df.groupby(df["date"].dt.date)["sentiment_score"].mean()
    spikes = daily_sentiment[daily_sentiment < threshold]
    return spikes

# Main Streamlit app
def main():
    st.title("Startup Brand Sentiment Analyzer")
    st.markdown("Monitor and analyze brand sentiment across social media and reviews.")

    # Sidebar for user inputs
    st.sidebar.header("Settings")
    brand = st.sidebar.text_input("Brand Name", value="MyStartup")
    platforms = st.sidebar.multiselect("Platforms", ["X", "Yelp"], default=["X", "Yelp"])
    days = st.sidebar.slider("Lookback Period (Days)", 1, 30, 7)
    negative_threshold = st.sidebar.slider("Negative Sentiment Threshold", -1.0, 0.0, -0.5, step=0.1)

    if st.sidebar.button("Analyze Sentiment"):
        with st.spinner("Scraping and analyzing data..."):
            # Scrape and analyze data
            df = scrape_social_media(brand, platforms, days)
            df = analyze_sentiment(df)

            # Display raw data
            st.subheader("Raw Mentions")
            st.dataframe(df[["date", "platform", "text", "sentiment", "sentiment_score"]])

            # Sentiment trends by platform
            st.subheader("Sentiment Trends by Platform")
            fig_platform = plot_sentiment_trends(df, "platform")
            st.plotly_chart(fig_platform, use_container_width=True)

            # Sentiment trends by date
            st.subheader("Overall Sentiment Trend")
            fig_date = plot_sentiment_trends(df, "platform")
            st.plotly_chart(fig_date, use_container_width=True)

            # Negative sentiment alerts
            st.subheader("Negative Sentiment Alerts")
            spikes = detect_negative_spikes(df, negative_threshold)
            if not spikes.empty:
                st.error("Negative sentiment spikes detected!")
                for date, score in spikes.items():
                    st.write(f"Date: {date}, Average Sentiment Score: {score:.2f}")
            else:
                st.success("No negative sentiment spikes detected.")

            # Marketing campaign integration
            st.subheader("Marketing Campaign Integration")
            if spikes.empty:
                st.write("Sentiment is stable. No immediate campaign adjustments needed.")
            else:
                st.warning("Negative sentiment detected. Consider adjusting campaigns.")
                if st.button("Generate Campaign Suggestions (AI Advertising Writer)"):
                    st.write("Connecting to AI Advertising Writer...")
                    st.write("Suggested action: Launch a positive PR campaign highlighting customer success stories.")

# Run the app
if __name__ == "__main__":
    main()
