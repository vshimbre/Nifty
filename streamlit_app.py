import streamlit as st
import requests
import pandas as pd
from textblob import TextBlob
from bs4 import BeautifulSoup

# --- Streamlit UI ---
st.set_page_config(page_title="NIFTY Option Chain Analysis", layout="wide")

st.title("📈 NIFTY Market Analysis & Prediction")
st.markdown("Real-time option chain analysis & market movement prediction for NIFTY.")

# --- Fetch NIFTY Option Chain ---
def fetch_option_chain():
    try:
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nseindia.com/",
            "accept": "application/json",
        }

        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)  # Session to avoid blocking
        response = session.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            
            # Debug: Print API response structure
            st.write("🔍 API Response Sample:", data)

            records = data.get("records", {}).get("data", [])
            if not records:
                st.error("❌ No option chain data found in API response.")
                return pd.DataFrame()

            extracted_data = []
            for record in records:
                extracted_data.append({
                    "strikePrice": record.get("strikePrice"),
                    "expiryDate": record.get("expiryDate"),
                    "CE_openInterest": record.get("CE", {}).get("openInterest", 0),
                    "PE_openInterest": record.get("PE", {}).get("openInterest", 0),
                    "CE_lastPrice": record.get("CE", {}).get("lastPrice", 0),
                    "PE_lastPrice": record.get("PE", {}).get("lastPrice", 0),
                })

            df = pd.DataFrame(extracted_data)
            st.write("🔍 Extracted Option Chain Data:", df.head())  # Debugging

            return df

        st.error(f"❌ NSE API returned status code {response.status_code}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ NSE API error: {e}")
        return pd.DataFrame()

# --- Fetch News & Perform Sentiment Analysis ---
def fetch_news_sentiment():
    try:
        url = "https://www.moneycontrol.com/news/business/markets/"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            headlines = [h.text for h in soup.select(".headline")[:5]]  # Fetch top 5 headlines
            
            sentiment_score = sum([TextBlob(headline).sentiment.polarity for headline in headlines]) / len(headlines)
            sentiment = "Bullish" if sentiment_score > 0 else "Bearish" if sentiment_score < 0 else "Neutral"
            
            return sentiment, headlines
        else:
            return "Neutral", ["❌ Failed to fetch news"]
    except Exception as e:
        return "Neutral", [f"❌ Error fetching news: {e}"]

# --- Predict Market Movement ---
def predict_market_trend(option_chain):
    if option_chain.empty:
        return "❌ Insufficient Data for Prediction"

    call_oi = option_chain["CE_openInterest"].sum()
    put_oi = option_chain["PE_openInterest"].sum()
    sentiment, _ = fetch_news_sentiment()

    if put_oi > call_oi and sentiment == "Bullish":
        return "📈 Up (Bullish)"
    elif call_oi > put_oi and sentiment == "Bearish":
        return "📉 Down (Bearish)"
    else:
        return "➖ Neutral"

# --- Predict Target Price ---
def predict_target_price(nifty_price, option_chain):
    if option_chain.empty:
        return "❌ Insufficient Data for Prediction"

    call_oi = option_chain["CE_openInterest"].sum()
    put_oi = option_chain["PE_openInterest"].sum()
    sentiment, _ = fetch_news_sentiment()

    if put_oi > call_oi and sentiment == "Bullish":
        target_price = nifty_price * 1.01  # Predict 1% increase
    elif call_oi > put_oi and sentiment == "Bearish":
        target_price = nifty_price * 0.99  # Predict 1% decrease
    else:
        target_price = nifty_price  # No change expected

    return round(target_price, 2)

# --- Main Execution ---
option_chain = fetch_option_chain()
st.subheader("📊 Option Chain Data")
st.dataframe(option_chain.head() if not option_chain.empty else "❌ Failed to fetch data")

# --- Market Prediction ---
st.subheader("📉 Market Prediction (Next 15 Minutes)")
prediction = predict_market_trend(option_chain)
st.write(f"**Predicted Market Move:** {prediction}")

# --- Target Price Prediction ---
nifty_price = 22000  # Set manually or fetch from API
target_price = predict_target_price(nifty_price, option_chain)
st.write(f"🎯 **Target Price (Next 15 min):** {target_price}")

# --- News Sentiment ---
st.subheader("📰 News Sentiment Analysis")
sentiment, headlines = fetch_news_sentiment()
st.write(f"**Market Sentiment:** {sentiment}")
st.write("**Top Headlines:**")
for headline in headlines:
    st.markdown(f"- {headline}")
