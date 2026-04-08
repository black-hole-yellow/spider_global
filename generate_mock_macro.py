import pandas as pd
import json
import numpy as np

def generate_historical_macro():
    """
    Floods your macro_events.json with realistic release dates 
    (e.g., NFP on the first Friday of the month) spanning 3 years.
    """
    dates = pd.date_range(start="2023-01-01", end="2026-01-01", freq="B")
    events = []
    
    for d in dates:
        # First Friday = NFP
        if d.weekday() == 4 and d.day <= 7:
            events.append({
                "date": f"{d.strftime('%Y-%m-%d')}T13:30:00Z", 
                "event": "US NFP", 
                "surprise": float(np.random.choice([-50000, 50000]))
            })
        # Second Tuesday = US and UK CPI
        elif d.weekday() == 1 and 8 <= d.day <= 14:
            events.append({
                "date": f"{d.strftime('%Y-%m-%d')}T13:30:00Z", 
                "event": "US CPI", 
                "surprise": float(np.random.choice([-0.2, 0.2]))
            })
            events.append({
                "date": f"{d.strftime('%Y-%m-%d')}T07:00:00Z", 
                "event": "UK CPI", 
                "surprise": float(np.random.choice([-0.2, 0.2]))
            })
        # Third Wednesday = FOMC & BoE
        elif d.weekday() == 2 and 15 <= d.day <= 21:
            events.append({
                "date": f"{d.strftime('%Y-%m-%d')}T19:00:00Z", 
                "event": "FOMC Rate Decision", 
                "surprise": 0.0
            })
            events.append({
                "date": f"{d.strftime('%Y-%m-%d')}T12:00:00Z", 
                "event": "BoE Rate Decision", 
                "surprise": 0.0
            })
            
    with open("data/macro_events.json", "w") as f:
        json.dump(events, f, indent=4)
        
    print(f"✅ Flooded system with {len(events)} historical macro events!")

if __name__ == "__main__":
    generate_historical_macro()