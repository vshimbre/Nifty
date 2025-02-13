import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Sentiment Analysis Model (FinBERT)
news_sentiment_model = pipeline("text-classification", model="ProsusAI/finbert", return_all_scores=True)

# Fetch NIFTY Live Price
def get_nifty_price():
    stock = yf.Ticker("^NSEI")
    data = stock.history(period="1d", interval="5m")
    return data.iloc[-1]["Close"] if not data.empty else None

# Fetch NIFTY Option Chain (Web Scraping)
def fetch_option_chain():
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
    return None

# Fetch Market News
def get_latest_news():
    url = "https://www.moneycontrol.com/news/business/markets/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    headlines = [a.text.strip() for a in soup.find_all("h2")[:5]]
    return headlines

# Analyze News Sentiment
def analyze_news_sentiment():
    news_list = get_latest_news()
    scores = [news_sentiment_model(news)[0] for news in news_list]
    avg_sentiment = sum([score[0]['score'] if score[0]['label'] == "positive" else -score[0]['score'] for score in scores]) / len(scores)
    return "Bullish" if avg_sentiment > 0.05 else "Bearish" if avg_sentiment < -0.05 else "Neutral"

# Predict Market Movement Based on Option Chain & News Sentiment
def predict_market_trend(option_chain, news_sentiment):
    if option_chain is None:
        return "❌ Option Chain Data Not Available"
    
    # Simple strategy: Check PCR (Put-Call Ratio)
    pcr = option_chain["PUT OI"].sum() / option_chain["CALL OI"].sum()
    
    # Decision based on PCR & Sentiment
    if pcr > 1 and news_sentiment == "Bullish":
        return "Up"
    elif pcr < 1 and news_sentiment == "Bearish":
        return "Down"
    else:
        return "Neutral"

# Streamlit UI
st.title("📈 NIFTY Option Chain & Market Prediction")

# Live NIFTY Price
st.subheader("Live NIFTY Price")
nifty_price = get_nifty_price()
st.write(f"**Current NIFTY Price:** {nifty_price}")

# Option Chain Data
st.subheader("📊 Option Chain Data")
option_chain = fetch_option_chain()
st.dataframe(option_chain.head() if option_chain is not None else "❌ Failed to fetch data")

# Market News & Sentiment
st.subheader("📰 Market News & Sentiment Analysis")
news_list = get_latest_news()
news_sentiment = analyze_news_sentiment()
st.write(f"**News Sentiment:** {news_sentiment}")
for news in news_list:
    st.write(f"- {news}")

# Market Prediction (Next 15 Min)
st.subheader("📉 Market Prediction (Next 15 Minutes)")
prediction = predict_market_trend(option_chain, news_sentiment)
st.write(f"**Predicted Market Move:** 🚀 {prediction}")
