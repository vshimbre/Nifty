import streamlit as st
import requests
import pandas as pd
from textblob import TextBlob
from bs4 import BeautifulSoup
import yfinance as yf  # For fetching NIFTY price
import plotly.graph_objects as go

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
        session.get("https://www.nseindia.com", headers=headers)
        response = session.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()

            records = data.get("records", {}).get("data", [])
            if not records:
                st.error("❌ No option chain data found in API response.")
                return pd.DataFrame()

            extracted_data = []
            for record in records:
                try:  # Handle potential missing data within each record
                    extracted_data.append({
                        "strikePrice": record.get("strikePrice"),
                        "expiryDate": record.get("expiryDate"),
                        "CE_openInterest": record.get("CE", {}).get("openInterest", 0),
                        "PE_openInterest": record.get("PE", {}).get("openInterest", 0),
                        "CE_lastPrice": record.get("CE", {}).get("lastPrice", 0),
                        "PE_lastPrice": record.get("PE", {}).get("lastPrice", 0),
                    })
                except (KeyError, TypeError) as e:
                    st.warning(f"⚠️ Issue parsing record: {record}. Error: {e}")  # Handle gracefully

            df = pd.DataFrame(extracted_data)
            return df
        else:
            st.error(f"❌ NSE API returned status code {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ NSE API error: {e}")
        return pd.DataFrame()

# --- Fetch News & Perform Sentiment Analysis ---
def fetch_news_sentiment():
    try:
        url = "https://www.moneycontrol.com/news/business/markets/"  # Or another reliable news source
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            headlines = [h.text for h in soup.select(".headline")[:5]] # Adjust selector as needed

            if headlines: # Handle case when no headlines are found
                sentiment_score = sum([TextBlob(headline).sentiment.polarity for headline in headlines]) / len(headlines)
                sentiment = "Bullish" if sentiment_score > 0.1 else "Bearish" if sentiment_score < -0.1 else "Neutral" # Adjust thresholds
                return sentiment, headlines
            else:
                return "Neutral", ["❌ No headlines found."]

        else:
            return "Neutral", ["❌ Failed to fetch news"]
    except Exception as e:
        return "Neutral", [f"❌ Error fetching news: {e}"]

# --- Fetch NIFTY Price ---
def fetch_nifty_price():
    try:
        nifty = yf.Ticker("^NSEI") # Use yfinance for NIFTY
        data = nifty.history(period="1d") # Get 1 day of historical data
        if not data.empty:
            return data['Close'][-1] # Return the closing price
        else:
            st.error("❌ Could not fetch NIFTY price from yfinance.")
            return None

    except Exception as e:
        st.error(f"❌ Error fetching NIFTY price: {e}")
        return None



# --- Predict Market Movement (Improved) ---
def predict_market_trend(option_chain, nifty_price):
    if option_chain.empty or nifty_price is None:
        return "❌ Insufficient Data for Prediction"

    call_oi = option_chain["CE_openInterest"].sum()
    put_oi = option_chain["PE_openInterest"].sum()
    sentiment, _ = fetch_news_sentiment()

    # More nuanced logic (example)
    if put_oi > call_oi * 1.2 and sentiment == "Bullish" and nifty_price > 19000: # Example thresholds
        return "📈 Up (Bullish)"
    elif call_oi > put_oi * 1.2 and sentiment == "Bearish" and nifty_price < 20000: # Example thresholds
        return "📉 Down (Bearish)"
    else:
        return "➖ Neutral"


# --- Predict Target Price (Improved - Example) ---
def predict_target_price(nifty_price, option_chain):
    if option_chain.empty or nifty_price is None:
        return "❌ Insufficient Data for Prediction"

    # Example: Simple moving average prediction (replace with more sophisticated method)
    # (Requires historical data, which is not included in this example)

    # Placeholder - Replace with your actual prediction logic
    target_price = nifty_price  # Default: No change

    return round(target_price, 2)

# --- Main Execution ---
nifty_price = fetch_nifty_price()  # Fetch real-time NIFTY price

if nifty_price is not None:
    st.write(f"**Current NIFTY Price:** {nifty_price}")

    option_chain = fetch_option_chain()

    if not option_chain.empty:
        st.subheader("📊 Option Chain Data")
        st.dataframe(option_chain)

        # Calculate Put-Call Ratio
        option_chain['PCR'] = option_chain['PE_openInterest'] / (option_chain['CE_openInterest'] + 1e-6) # prevent divide by zero
        st.write("Put-Call Ratio:", option_chain['PCR'].sum())

        # --- Market Prediction ---
        st.subheader("📉 Market Prediction (Next 15 Minutes)")
        prediction = predict_market_trend(option_chain, nifty_price)
        st.write(f"**Predicted Market Move:** {prediction}")

        # --- Target Price Prediction ---
        target_price = predict_target_price(nifty_price, option_chain)
        st.write(f"🎯 **Target Price (Next 15 min):** {target_price}")

        # --- News Sentiment ---
        st.subheader("📰 News Sentiment Analysis")
        sentiment, headlines = fetch_news_sentiment()
        st.write(f"**Market Sentiment:** {sentiment}")
        st.write("**Top Headlines:**")
        for headline in headlines:
            st.markdown(f"- {headline}")

    else:
        st.write("Could not fetch option chain data.")

