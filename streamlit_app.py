import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Sentiment Analysis Model
news_sentiment_model = pipeline("text-classification", model="ProsusAI/finbert", return_all_scores=True)

# Fetch NIFTY Live Price
def get_nifty_price():
    try:
        stock = yf.Ticker("^NSEI")
        data = stock.history(period="1d", interval="5m")
        return data.iloc[-1]["Close"] if not data.empty else None
    except Exception as e:
        st.error(f"❌ Error fetching NIFTY price: {e}")
        return None

# Fetch NIFTY Option Chain (Web Scraping)
def fetch_option_chain():
    try:
        url = "https://www.nseindia.com/option-chain"
        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        session.get(url, headers=headers)

        response = session.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"id": "optionChainTable"})

        if table:
            df = pd.read_html(str(table))[0]
            return df

        return pd.DataFrame()  # Return empty DataFrame if data is not found
    except Exception as e:
        st.error(f"❌ Error fetching option chain: {e}")
        return pd.DataFrame()  # Return empty DataFrame to avoid errors

# Fetch Market News
def get_latest_news():
    try:
        url = "https://www.moneycontrol.com/news/business/markets/"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = [a.text.strip() for a in soup.find_all("h2")[:5]]
        return headlines
    except Exception as e:
        st.error(f"❌ Error fetching news: {e}")
        return []

# Analyze News Sentiment
def analyze_news_sentiment():
    news_list = get_latest_news()
    if not news_list:
        return "Neutral"

    scores = [news_sentiment_model(news)[0] for news in news_list]
    avg_sentiment = sum([score[0]['score'] if score[0]['label'] == "positive" else -score[0]['score'] for score in scores]) / len(scores)

    return "Bullish" if avg_sentiment > 0.05 else "Bearish" if avg_sentiment < -0.05 else "Neutral"

# Predict Market Movement
def predict_market_trend():
    option_chain = fetch_option_chain()
    
    if option_chain.empty:
        return "❌ Insufficient Data for Prediction"
    
    # Extract Call & Put Open Interest (OI)
    call_oi = option_chain["OI"].iloc[:10].sum()  # Sum of top 10 Call OI
    put_oi = option_chain["OI"].iloc[-10:].sum()  # Sum of top 10 Put OI
    
    news_sentiment = analyze_news_sentiment()

    # Combine Sentiment & OI Data
    if put_oi > call_oi and news_sentiment == "Bullish":
        return "📈 Up"
    elif call_oi > put_oi and news_sentiment == "Bearish":
        return "📉 Down"
    else:
        return "🔄 Neutral"

# Streamlit UI
st.title("📈 NIFTY Option Chain & Market Prediction")

# Live NIFTY Price
st.subheader("Live NIFTY Price")
nifty_price = get_nifty_price()
if nifty_price:
    st.write(f"**Current NIFTY Price:** {nifty_price}")
else:
    st.warning("⚠ Could not fetch live price.")

# Option Chain Data
st.subheader("📊 Option Chain Data")
option_chain = fetch_option_chain()

if option_chain.empty:
    st.warning("❌ Option Chain data not available.")
else:
    st.dataframe(option_chain.head())  # Display first few rows

# Market News & Sentiment
st.subheader("📰 Market News & Sentiment Analysis")
news_list = get_latest_news()
news_sentiment = analyze_news_sentiment()
st.write(f"**News Sentiment:** {news_sentiment}")

if news_list:
    for news in news_list:
        st.write(f"- {news}")
else:
    st.warning("⚠ No news headlines found.")

# Market Prediction (Next 15 Min)
st.subheader("📉 Market Prediction (Next 15 Minutes)")
prediction = predict_market_trend()
st.write(f"**Predicted Market Move:** {prediction}")
