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

# Fetch NIFTY Option Chain (Using NSE API)
def fetch_option_chain():
    try:
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nseindia.com/",
            "accept": "application/json",
        }

        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)  # Create session to avoid blocking
        response = session.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            records = data.get("records", {}).get("data", [])

            if not records:
                st.error("❌ No option chain data found in API response.")
                return pd.DataFrame()

            df = pd.DataFrame(records)

            # Print column names for debugging
            st.write("🔍 Option Chain Columns:", df.columns.tolist())

            return df

        st.error(f"❌ NSE API returned status code {response.status_code}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ NSE API error: {e}")
        return pd.DataFrame()

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
    
    # Print columns to check actual available names
    st.write("📊 Available Option Chain Columns:", option_chain.columns.tolist())

    # Identify correct column names
    call_oi_col = next((col for col in option_chain.columns if "CE_openInterest" in col or "CE" in col.lower()), None)
    put_oi_col = next((col for col in option_chain.columns if "PE_openInterest" in col or "PE" in col.lower()), None)

    if not call_oi_col or not put_oi_col:
        return "❌ Missing Open Interest Data in API Response"

    # Extract Call & Put Open Interest (OI)
    call_oi = option_chain[call_oi_col].iloc[:10].sum()
    put_oi = option_chain[put_oi_col].iloc[:10].sum()
    
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
