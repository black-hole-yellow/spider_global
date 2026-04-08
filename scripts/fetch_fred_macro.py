import pandas as pd
import json
import os

def fetch_fred_series_local(ticker):
    """Читает данные FRED из локальных CSV файлов."""
    filepath = f"data/raw/fred/{ticker}.csv"
    
    print(f"Чтение {ticker} из {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Ищем колонку с датой (в твоем случае это 'observation_date')
    date_col = next((c for c in df.columns if 'DATE' in c.upper()), None)
    if date_col is None:
        # Если почему-то не нашли, берем просто первую колонку
        date_col = df.columns[0]
    
    # Парсим даты и ставим в индекс
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.index.name = 'DATE' # Приводим к единому имени для всех 3-х файлов
    
    # Переводим в числа
    df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
    return df

def fetch_and_synthesize_macro_events():
    print("1. Загрузка локальных данных...")
    tickers = ['UKUN', 'UKCPI', 'UKINRATE']
    
    try:
        # Читаем все три файла
        dfs = [fetch_fred_series_local(t) for t in tickers]
        # Склеиваем их по дате
        df = pd.concat(dfs, axis=1)
        
        # Сортируем индекс перед срезом (обязательно для Pandas)
        df.sort_index(inplace=True)
        # Оставляем только данные с 2000 года
        df = df.loc['2000-01-01':]
        # Заполняем пропуски предыдущими значениями (Forward Fill)
        df = df.ffill() 
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return

    # Считаем разницу между месяцами
    df['FED_Change'] = df['FEDFUNDS'].diff()
    df['CPI_Change'] = df['CPIAUCSL'].pct_change() * 100
    df['UNRATE_Change'] = df['UNRATE'].diff()
    
    events = []
    print("2. Синтез текстовых новостей (Feature Generation)...")
    
    for index, row in df.iterrows():
        if pd.isna(row['FEDFUNDS']): continue
        date_str = index.strftime('%Y-%m-%d 14:00:00') 
        
        if pd.notna(row['FED_Change']) and row['FED_Change'] != 0:
            direction = "increased" if row['FED_Change'] > 0 else "decreased"
            tone = "hawkish" if row['FED_Change'] > 0 else "dovish"
            events.append({
                "timestamp": date_str,
                "title": f"Federal Reserve {direction} Interest Rate",
                "content": f"The US Federal Reserve {direction} the key interest rate to {row['FEDFUNDS']}%. This is a {tone} move.",
                "source": "FRED_Synthetic",
                "impact_currency": "USD"
            })
        
        if pd.notna(row['CPI_Change']) and abs(row['CPI_Change']) > 0.1:
            dir_cpi = "rose" if row['CPI_Change'] > 0 else "fell"
            events.append({
                "timestamp": index.strftime('%Y-%m-%d 08:30:00'),
                "title": f"US CPI {dir_cpi}",
                "content": f"US monthly inflation {dir_cpi} by {abs(row['CPI_Change']):.2f}%.",
                "source": "FRED_Synthetic",
                "impact_currency": "USD"
            })
            
        if pd.notna(row['UNRATE_Change']) and abs(row['UNRATE_Change']) >= 0.1:
            dir_unrate = "jumped" if row['UNRATE_Change'] > 0 else "dropped"
            events.append({
                "timestamp": index.strftime('%Y-%m-%d 08:30:00'),
                "title": f"US Unemployment Rate {dir_unrate}",
                "content": f"The US unemployment rate {dir_unrate} to {row['UNRATE']}%.",
                "source": "FRED_Synthetic",
                "impact_currency": "USD"
            })

    return events

def save_events(events, filepath="data/macro_events.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=4, ensure_ascii=False)
    print(f"\n✅ ГОТОВО! Успешно сохранено {len(events)} событий в {filepath}!")

if __name__ == "__main__":
    events = fetch_and_synthesize_macro_events()
    if events:
        save_events(events)