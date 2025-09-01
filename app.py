from flask import Flask, jsonify, send_from_directory
import threading
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import joblib
import os
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import csv
# Remove the hardcoded sys.path line
from predictor import forecast_prices_noloop

# Configuration - Updated to use current directory
BASE_DIR = os.getcwd()  # Current working directory
NEWS_CSV_PATH = os.path.join(BASE_DIR, 'news1-Copy.csv')
BASE_URL = "https://newsapi.org/v2/everything"
API_KEY = 'cb0e07a51f4a405590d9e324f6e3b309'
SEQUENCE_LEN = 20

# Global flag to track prediction status
prediction_running = False

#lexicon Patching
def load_custom_lexicon(file_path):
    lexicon = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            term, score = line.strip().split('\t')
            lexicon[term] = float(score)
    return lexicon

app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')

predicted_data = []

# Updated model and scaler paths
model = load_model(os.path.join(BASE_DIR, "USD_Price_Forecst_Model_many_curr_sentiment_hb.h5"))
scaler = joblib.load(os.path.join(BASE_DIR, "input_scaler_many_curr_setiment_hb.save"))
target_scaler = joblib.load(os.path.join(BASE_DIR, "output_scaler_many_curr_sentiment_hb.save"))

# Updated CSV paths
df = pd.read_csv(os.path.join(BASE_DIR, 'USD_INR Historical Data (1).csv'))

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Change %'] = df['Change %'].str.replace('%', '').astype(float)

df['Sentiment'] = 0.0

for idx, row in df.iterrows():
    change = max(min(row['Change %'], 1), -1)
    base = (change / 1) * 0.3
    noise = random.uniform(-0.02, 0.02)
    scaled_sentiment = round(base + noise, 3)
    df.at[idx, 'Sentiment'] = scaled_sentiment

analyzer = SentimentIntensityAnalyzer()
finance_lexicon = load_custom_lexicon("finance_lexicon.txt")
analyzer.lexicon.update(finance_lexicon)

if not os.path.exists(NEWS_CSV_PATH):
    with open(NEWS_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['publishedAt', 'title', 'source'])

news_df = pd.read_csv(NEWS_CSV_PATH)
if len(news_df) > 0:
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt']).dt.date

# Prepare historical data with sentiment from news (for first 15 rows)
for idx in df.head(15).index:
    date = df.at[idx, 'Date']
    from_date = date - timedelta(days=1)
    to_date = date + timedelta(days=1)
    
    params = {
        'qInTitle': 'RBI OR USD OR INR OR Indian Rupee OR Forex',
        'sources': 'the-hindu,bloomberg,cnbc,financial-times,economic-times',
        'from': from_date.isoformat(),
        'to': to_date.isoformat(),
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 5,
        'apiKey': API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    sentiments = []
    
    if 'articles' in data:
        for article in data['articles']:
            title = article['title']
            score = analyzer.polarity_scores(title)['compound']
            sentiments.append(score)
    
    avg_sentiment = round(np.mean(sentiments), 3) if sentiments else 0.0
    df.at[idx, 'Sentiment'] = avg_sentiment

df = df[::-1].reset_index(drop=True)

rolling_df = df[['Date', 'Price', 'Change %', 'Sentiment']].copy()

print(f"this is 1st rolling_df: {rolling_df.head(5)}")

# Updated currency data paths
jpy1 = pd.read_csv(os.path.join(BASE_DIR, "USD_JPY Historical Data.csv"))
gbp1 = pd.read_csv(os.path.join(BASE_DIR, "USD_GBP Historical Data.csv"))
eur1 = pd.read_csv(os.path.join(BASE_DIR, "USD_EUR Historical Data (1).csv"))

jpy1 = jpy1[::-1].reset_index(drop=True)
gbp1 = gbp1[::-1].reset_index(drop=True)
eur1 = eur1[::-1].reset_index(drop=True)

@app.route('/')
def serve_html():
    return send_from_directory(BASE_DIR, 'interface_try_reponsive.html')

@app.route('/<path:filename>')
def serve_files(filename):
    return send_from_directory(BASE_DIR, filename)

@app.route('/latest-price')
def latest_price():
    return jsonify(predicted_data[-30:])

# New endpoints for manual control
@app.route('/start-predictions')
def start_predictions():
    global prediction_running
    if not prediction_running:
        prediction_running = True
        threading.Thread(target=prediction_loop_with_24h_cycle, daemon=True).start()
        return jsonify({
            'status': 'started', 
            'message': 'Predictions started with 24-hour cycles',
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'status': 'already_running', 
            'message': 'Predictions already in progress',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/stop-predictions')
def stop_predictions():
    global prediction_running
    prediction_running = False
    return jsonify({
        'status': 'stopped', 
        'message': 'Predictions stopped',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/status')
def get_status():
    return jsonify({
        'app_status': 'running',
        'prediction_running': prediction_running,
        'predictions_count': len(predicted_data),
        'last_prediction': predicted_data[-1]['date'] if predicted_data else None,
        'timestamp': datetime.now().isoformat()
    })

def prediction_loop_with_24h_cycle():
    """Modified prediction loop that runs 60 predictions, then sleeps for 24 hours"""
    global rolling_df, news_df, jpy1, gbp1, eur1, prediction_running
    
    cycle_count = 1
    
    while prediction_running:
        print(f"\n[INFO] Starting prediction cycle {cycle_count} at {datetime.now()}")
        
        # Run 60 predictions in this cycle
        for i in range(60):
            if not prediction_running:  # Check if stopped during cycle
                print("[INFO] Predictions stopped by user")
                return
                
            print(f"\n[INFO] Prediction iteration {i + 1}/60 (Cycle {cycle_count})")
            
            # Prepare features
            rolling_df['day_of_week'] = rolling_df['Date'].dt.weekday
            rolling_df['month'] = rolling_df['Date'].dt.month
            rolling_df['delta_price'] = rolling_df['Price'].diff()
            rolling_df['price_pct'] = rolling_df['Price'].pct_change()
            rolling_df['rolling_mean_3'] = rolling_df['Price'].rolling(3).mean()
            rolling_df['volatility_3'] = rolling_df['Price'].rolling(3).std()
            rolling_df['euro'] = eur1['Price'].iloc[-1]
            rolling_df['pound'] = gbp1['Price'].iloc[-1]
            rolling_df['yen'] = jpy1['Price'].iloc[-1]
            
            feature_cols = ['Price', 'day_of_week', 'month','Sentiment', 
                            'delta_price', 'price_pct', 'rolling_mean_3', 'volatility_3',
                            'euro', 'pound', 'yen']
            
            rolling_df = rolling_df.dropna()
            
            if len(rolling_df) < SEQUENCE_LEN:
                print(f"[ERROR] Insufficient data: {len(rolling_df)} < {SEQUENCE_LEN}")
                break
            
            latest_sequence = rolling_df[feature_cols].tail(SEQUENCE_LEN)
            latest_scaled = scaler.transform(latest_sequence)
            latest_scaled = np.expand_dims(latest_scaled, axis=0)
            
            change_scaled = model.predict(latest_scaled)
            predicted_change = target_scaler.inverse_transform(change_scaled)[0][0]
            
            last_row = rolling_df.iloc[-1]
            last_price = last_row['Price']
            next_date = last_row['Date'] + timedelta(days=1)
            
            # Get currency values
            last_euro = last_row['euro']
            last_pound = last_row['pound']
            last_yen = last_row['yen']
            
            jpy1, change_jpy, new_price_jpy,prev_price_jpy = forecast_prices_noloop(jpy1, model_name="JPY_Price_Forecst_Model")
            gbp1, change_gbp, new_price_gbp,prev_price_gbp = forecast_prices_noloop(gbp1, model_name="GBP_Price_Forecst_Model")
            eur1, change_eur, new_price_eur,prev_price_eur = forecast_prices_noloop(eur1, model_name="EUR_Price_Forecst_Model")
            
            last_euro = prev_price_eur
            last_pound = prev_price_gbp
            last_yen = prev_price_jpy

            updated_price = last_price + predicted_change
            updated_euro = new_price_eur
            updated_pound = new_price_gbp
            updated_yen = new_price_jpy
            
            date_str = next_date.strftime("%Y-%m-%d")

            # Check if news exists for this date in CSV
            headlines = []
            if len(news_df) > 0 and next_date.date() in news_df['publishedAt'].values:
                # Get news from CSV
                daily_news = news_df[news_df['publishedAt'] == next_date.date()]
                sentiments = []
                
                for _, article_row in daily_news.head(3).iterrows():
                    title = article_row['title']
                    sentiment = analyzer.polarity_scores(title)['compound']
                    sentiments.append(sentiment)
                    
                    headlines.append({
                        'title': title,
                        'source': article_row['source'],
                        'time': str(article_row['publishedAt'])
                    })
                
                avg_sentiment = -((sum(sentiments) / len(sentiments))) if sentiments else 0.0
                
            else:
                # Fetch new news from API
                params = {
                    'qInTitle': 'RBI OR USD OR INR OR Indian Rupee OR Forex',
                    'sources': 'the-hindu,bloomberg,cnbc,financial-times,economic-times',
                    'from': date_str,
                    'to': date_str,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 5,
                    'apiKey': API_KEY
                }
                
                response = requests.get(BASE_URL, params=params)
                data = response.json()
                sentiments = []
                
                # Process articles and save to CSV
                if 'articles' in data:
                    new_rows = []
                    for article in data['articles'][:3]:
                        title = article['title']
                        source = article.get('source', {}).get('name', 'Unknown')
                        published_at = next_date.date()
                        
                        sentiment = analyzer.polarity_scores(title)['compound']
                        sentiments.append(sentiment)
                        
                        headlines.append({
                            'title': title,
                            'source': source,
                            'time': article.get('publishedAt', '')[:16].replace('T', ' ')
                        })
                        
                        new_rows.append([published_at, title, source])
                    
                    # Append to CSV
                    if new_rows:
                        with open(NEWS_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerows(new_rows)
                        
                        # Update news_df
                        new_df = pd.DataFrame(new_rows, columns=['publishedAt', 'title', 'source'])
                        new_df['publishedAt'] = pd.to_datetime(new_df['publishedAt']).dt.date
                        news_df = pd.concat([news_df, new_df], ignore_index=True)
                
                avg_sentiment = -((sum(sentiments) / len(sentiments))) if sentiments else 0.0
            
            print(f"[INFO] Calculated Sentiment {avg_sentiment:.3f} for date {date_str}")
            
            new_row = {
                'Date': next_date,
                'Price': updated_price,
                'Change %': predicted_change,
                'Sentiment': avg_sentiment,
                'day_of_week': next_date.weekday(),
                'month': next_date.month,
                'delta_price': predicted_change,
                'price_pct': predicted_change / last_price,
                'rolling_mean_3': rolling_df['Price'].tail(3).mean(),
                'volatility_3': rolling_df['Price'].tail(3).std(),
                'euro': updated_euro,
                'pound': updated_pound,
                'yen': updated_yen
            }
            
            # Add new row to rolling_df
            rolling_df = pd.concat([rolling_df, pd.DataFrame([new_row])], ignore_index=True)

            deviation = updated_price - last_price
            trend = "Increased" if deviation > 0 else "Decreased"
            
            deviationeuro = updated_euro - last_euro
            trend1 = "Increased" if deviationeuro > 0 else "Decreased"
            
            deviationpound = updated_pound - last_pound
            trend2 = "Increased" if deviationpound > 0 else "Decreased"
            
            deviationyen = updated_yen - last_yen
            trend3 = "Increased" if deviationyen > 0 else "Decreased"
            
            # Store prediction data
            predicted_data.append({
                'date': next_date.strftime('%Y-%m-%d'),
                'price': float(round(updated_price, 4)),
                'deviation': float(round(deviation / last_price * 100, 2)),
                'status': trend,
                'euro': float(round(updated_euro, 2)),
                'deviationeuro': float(round(deviationeuro / last_euro * 100, 2)),
                'euro_status': trend1,
                'pound': float(round(updated_pound, 2)),
                'deviationpound': float(round(deviationpound / last_pound * 100, 2)),
                'pound_status': trend2,
                'yen': float(round(updated_yen, 2)),
                'deviationyen': float(round(deviationyen / last_yen * 100, 2)),
                'yen_status': trend3,
                'headlines': headlines,
                'volatility_3': float(rolling_df['Price'].tail(3).std())
            })
            
            # Print results
            print(f"[RESULT] {next_date.strftime('%Y-%m-%d')} → ₹{updated_price:.4f} | Deviation: ₹{deviation:.4f} | {trend}")
            print(f"[RESULT] EUR: {updated_euro:.6f} | GBP: {updated_pound:.6f} | JPY: {updated_yen:.4f}")
            
            time.sleep(2)
        
        # Complete cycle, now sleep for 24 hours
        if prediction_running:
            print(f"\n[INFO] Completed cycle {cycle_count} with 60 predictions")
            print(f"[INFO] Sleeping for 24 hours... Next cycle will start at {(datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Sleep for 24 hours (86400 seconds)
            sleep_time = 24 * 60 * 60
            start_sleep = time.time()
            
            while (time.time() - start_sleep) < sleep_time and prediction_running:
                time.sleep(60)  # Check every minute if we should stop
            
            cycle_count += 1
    
    print(f"[INFO] Prediction loop stopped at {datetime.now()}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Removed the automatic threading line - now controlled manually
    print(f"[INFO] App starting on port {port}")
    print(f"[INFO] Visit /start-predictions to begin prediction cycles")
    print(f"[INFO] Visit /status to check current status")
    app.run(host='0.0.0.0', port=port, use_reloader=False)
