import json
import os
from datetime import datetime

def generate_historical_macro_events():
    """
    Генерирует хардкод-базу реальных исторических макро-событий, 
    которые оказали огромное влияние на пару GBP/USD.
    Это необходимо для качественного бэктеста и проверки NLP-модели.
    """
    events = [
        # --- Эпоха Brexit и COVID (2020) ---
        {
            "timestamp": "2020-01-31 23:00:00",
            "title": "Brexit Officially Happens",
            "content": "The United Kingdom officially leaves the European Union. Transition period begins, but uncertainty over future trade deals with the EU weighs heavily on the British Pound.",
            "source": "Historical Data",
            "impact_currency": "GBP"
        },
        {
            "timestamp": "2020-03-15 17:00:00",
            "title": "Federal Reserve slashes rates to zero",
            "content": "The US Federal Reserve dramatically cuts interest rates to a range of 0% to 0.25% and launches a massive $700 billion quantitative easing program to shield the economy from the coronavirus pandemic. Massive USD volatility.",
            "source": "Historical Data",
            "impact_currency": "USD"
        },
        {
            "timestamp": "2020-03-19 12:00:00",
            "title": "Bank of England emergency rate cut",
            "content": "The Bank of England cuts its main interest rate to a record low of 0.1% and expands its bond-buying program by £200 billion to combat the economic shock of COVID-19.",
            "source": "Historical Data",
            "impact_currency": "GBP"
        },
        
        # --- Инфляционный шок и кризис (2022) ---
        {
            "timestamp": "2022-09-23 09:30:00",
            "title": "UK Government announces mini-budget",
            "content": "Chancellor Kwasi Kwarteng announces the biggest tax cuts in 50 years funded by massive government borrowing. Markets panic, causing a historic crash in UK Gilts and sending the Pound to an all-time low against the Dollar.",
            "source": "Historical Data",
            "impact_currency": "GBP"
        },
        {
            "timestamp": "2022-09-28 10:00:00",
            "title": "Bank of England intervenes in bond market",
            "content": "The Bank of England announces it will temporarily buy long-dated UK government bonds (Gilts) to restore orderly market conditions following the mini-budget fallout. GBP rebounds sharply.",
            "source": "Historical Data",
            "impact_currency": "GBP"
        },
        {
            "timestamp": "2022-11-02 14:00:00",
            "title": "Fed signals rates will stay higher for longer",
            "content": "Federal Reserve Chair Jerome Powell warns that the ultimate level of interest rates will be higher than previously expected to fight stubborn inflation, causing a broad USD rally.",
            "source": "Historical Data",
            "impact_currency": "USD"
        },

        # --- Период стабилизации (2023-2024) ---
        {
            "timestamp": "2023-08-03 12:00:00",
            "title": "BoE hikes rates to 15-year high",
            "content": "The Bank of England raises its key interest rate by 25 basis points to 5.25%, a 15-year high, but hints that borrowing costs may be nearing their peak. Pound shows mixed reaction.",
            "source": "Historical Data",
            "impact_currency": "GBP"
        },
        {
            "timestamp": "2023-12-13 14:00:00",
            "title": "Fed Pivot: Powell discusses rate cuts",
            "content": "The Federal Reserve holds rates steady, but the dot plot projections show policymakers expect 75 basis points of rate cuts in 2024. The US Dollar aggressively sells off against major peers including GBP.",
            "source": "Historical Data",
            "impact_currency": "USD"
        },
        {
            "timestamp": "2024-03-21 12:00:00",
            "title": "BoE holds rates, inflation cools",
            "content": "The Bank of England leaves interest rates unchanged at 5.25%. With UK inflation dropping faster than expected, markets begin pricing in summer rate cuts, putting downward pressure on the Pound.",
            "source": "Historical Data",
            "impact_currency": "GBP"
        }
    ]
    return events

def save_events_to_json(events, filepath="data/macro_events.json"):
    """Сохраняет список событий в JSON файл."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=4, ensure_ascii=False)
    
    print(f"Успешно сохранено {len(events)} исторических событий в {filepath}")

if __name__ == "__main__":
    print("Генерация исторической макро-базы для бэктестов...")
    historical_events = generate_historical_macro_events()
    save_events_to_json(historical_events)
    
    print("\nСовет: Теперь ты можешь запустить scripts/llm_macro_parser.py")
    print("Он превратит эти реальные исторические тексты в многомерные векторы (Embeddings)!")