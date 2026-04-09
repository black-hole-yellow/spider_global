import pandas as pd
import json
import os

def fetch_fred_series_local(ticker):
    """Читает данные из локальных CSV файлов."""
    filepath = f"data/raw/economic/{ticker}.csv"
    
    if not os.path.exists(filepath):
        print(f"⚠️ Файл не найден: {filepath} (Пропускаем)")
        return None
        
    df = pd.read_csv(filepath)
    
    # Ищем колонку с датой (Date, DATE, observation_date)
    date_col = next((c for c in df.columns if 'DATE' in c.upper()), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.index.name = 'DATE'
    
    # Ищем колонку со значениями (обычно это название тикера)
    val_col = next((c for c in df.columns if c.upper() == ticker.upper()), df.columns[0])
    df[ticker] = pd.to_numeric(df[val_col], errors='coerce')
    
    return df[[ticker]]

def generate_universal_macro_events():
    print("1. Загрузка локальных данных US и UK...")
    
    # Конфигурация: Тикер -> (Страна, Название показателя, Порог изменения)
    indicators = {
        'FEDFUNDS': ('US', 'Federal Reserve Interest Rate', 'USD', 0.0),
        'CPIAUCSL': ('US', 'CPI Inflation', 'USD', 0.1),
        'UNRATE':   ('US', 'Unemployment Rate', 'USD', 0.1),
        'UKINRATE': ('UK', 'Bank of England Interest Rate', 'GBP', 0.0),
        'UKCPI':    ('UK', 'CPI Inflation', 'GBP', 0.1),
        'UKUN':     ('UK', 'Unemployment Rate', 'GBP', 0.1)
    }
    
    dfs = []
    for ticker in indicators.keys():
        df_ticker = fetch_fred_series_local(ticker)
        if df_ticker is not None:
            dfs.append(df_ticker)
            
    if not dfs:
        print("❌ Не найдено ни одного CSV файла в data/raw/economic/")
        return []

    # Склеиваем все данные в единую таймлайн-таблицу
    df = pd.concat(dfs, axis=1)
    df.sort_index(inplace=True)
    df = df.loc['2000-01-01':].ffill() # Начинаем с 2000 года, заполняем пустоты
    
    events = []
    print("2. Синтез текстовых новостей (Feature Generation)...")
    
    # Проходим по таймлайну и генерируем новости при изменении показателей
    for ticker, (country, name, currency, threshold) in indicators.items():
        if ticker not in df.columns: continue
            
        # Для инфляции считаем процентное изменение, для остальных - абсолютное
        if 'CPI' in ticker:
            changes = df[ticker].pct_change() * 100
        else:
            changes = df[ticker].diff()
            
        for date, change in changes.dropna().items():
            if abs(change) > threshold:
                val = df.loc[date, ticker]
                
                # Формируем человекочитаемый текст
                direction = "increased" if change > 0 else "decreased"
                tone = "hawkish" if change > 0 else "dovish"
                
                # Время: Ставки обычно в 14:00 (US) / 12:00 (UK), Инфляция утром
                time_str = "14:00:00" if "RATE" in ticker or "FUNDS" in ticker else "08:30:00"
                timestamp_str = date.strftime(f'%Y-%m-%d {time_str}')
                
                events.append({
                    "timestamp": timestamp_str,
                    "title": f"{country} {name} {direction}",
                    "content": f"The {country} {name} {direction} to {val:.2f}%. This is considered a {tone} macroeconomic shift.",
                    "source": "Macro_Synthetic",
                    "impact_currency": currency
                })

    # Сортируем все события хронологически
    events.sort(key=lambda x: x["timestamp"])
    return events

def save_events(events, filepath="data/macro_events.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=4, ensure_ascii=False)
    print(f"\n✅ ГОТОВО! Успешно сохранено {len(events)} событий в {filepath}!")

if __name__ == "__main__":
    events = generate_universal_macro_events()
    if events:
        save_events(events)