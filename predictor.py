import joblib
from tensorflow.keras.models import load_model
from datetime import timedelta
import pandas as pd
import numpy as np
import time

# Make sure jpy is already defined

def forecast_prices_noloop(rolling_X_val_func, model_name="JPY_Price_Forecst_Model"):
    # Load model and scalers
    model = load_model(f"{model_name}.h5")
    scaler = joblib.load(f"{model_name}_input_scaler.save")
    target_scaler = joblib.load(f"{model_name}_output_scaler.save")

    # Ensure date is datetime
    rolling_X_val_func['Date'] = pd.to_datetime(rolling_X_val_func['Date'], dayfirst=True)
    rolling_X_val_func['Change %'] = (rolling_X_val_func['Change %'].astype(str).str.replace('%', '', regex=False).astype(float))

    # --- Step 1: Feature Engineering ---
    temp_df = rolling_X_val_func[['Date', 'Price', 'Change %']].copy()
    temp_df['day_of_week'] = temp_df['Date'].dt.weekday
    temp_df['month'] = temp_df['Date'].dt.month
    temp_df['delta_price'] = temp_df['Price'].diff()
    temp_df['price_pct'] = temp_df['Price'].pct_change()
    temp_df['rolling_mean_3'] = temp_df['Price'].rolling(3).mean()
    temp_df['volatility_3'] = temp_df['Price'].rolling(3).std()
    temp_df.dropna(inplace=True)

    # --- Step 2: Get latest 30 records ---
    current_sequence = temp_df.tail(30)
    X_input = current_sequence[['Price','day_of_week','month','Change %','delta_price','price_pct','rolling_mean_3','volatility_3']]
    X_scaled = scaler.transform(X_input).reshape(1, 30, 8)

    # --- Step 3: Predict ---
    y_pred = model.predict(X_scaled)
    predicted_change = target_scaler.inverse_transform(y_pred)[0][0]

    # --- Step 4: New row ---
    last_date = rolling_X_val_func['Date'].iloc[-1]
    next_date = last_date + timedelta(days=1)
    prev_price = rolling_X_val_func['Price'].iloc[-1]

    if model_name == "JPY_Price_Forecst_Model":
        new_price = prev_price + predicted_change
    else:
        new_price = prev_price * (1 + predicted_change / 100)

    new_row = {
        'Date': next_date,
        'Price': new_price,
        'Change %': predicted_change
    }

    # Append row
    rolling_X_val_func = pd.concat([rolling_X_val_func, pd.DataFrame([new_row])], ignore_index=True)

    #print(f"{next_date.strftime('%Y-%m-%d')} → ${new_price:.4f}")
    #print(f"The deviation of Price is {predicted_change:.2f}")
    #print("Price Increased" if new_price > prev_price else "Price Decreased")
    #print(f"Shape: {rolling_X_val_func.shape}")

    return rolling_X_val_func, predicted_change, new_price, prev_price
