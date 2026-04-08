import os
import json
import ollama

def run_local_alpha_audit():
    print("1. Загрузка ТОП-14 фичей после математического скрининга...")
    
    input_path = "data/processed/top_features.txt"
    if not os.path.exists(input_path):
        print(f"Ошибка: Файл {input_path} не найден.")
        return

    with open(input_path, 'r') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Загружено фичей для аудита: {len(features)}")
    print("2. Обращение к локальной модели через Ollama... (Это может занять пару минут)")
    
    prompt = f"""
    You are a Senior Quantitative Researcher at a Tier-1 Hedge Fund.
    We are building a Time-Series Transformer model to predict the 1-hour forward return of the GBP/USD currency pair.
    
    We have run a statistical screening and found that the following technical features have predictive power on historical data:
    {features}
    
    Your task is to evaluate the FUNDAMENTAL and MICROSTRUCTURAL logic of each feature for the GBP/USD pair.
    We want to avoid 'spurious correlations' (math without logic).
    
    For each feature, provide:
    1. A logic score from 1 to 10 (10 is absolute fundamental sense, 1 is pure mathematical noise/overfitting).
    2. A short, 1-sentence reasoning.

    You must respond ONLY with a JSON object containing an array called "audit", like this:
    {{
      "audit": [
        {{
          "feature_name": "string",
          "score": int,
          "reasoning": "string"
        }}
      ]
    }}
    """

    try:
        # Обращаемся к локальной Ollama
        response = ollama.chat(
            model='gemma4',
            messages=[
                {"role": "system", "content": "You are a quantitative finance expert. Output strict JSON only."},
                {"role": "user", "content": prompt}
            ],
            format='json' # Жестко заставляем модель выдавать JSON
        )
        
        result_text = response['message']['content']
        data = json.loads(result_text)
        audit_results = data.get("audit", [])
        
    except Exception as e:
        print(f"❌ Ошибка при обращении к Ollama или парсинге JSON: {e}")
        print("Ответ модели (если есть):", result_text if 'result_text' in locals() else "Нет ответа")
        return

    print("\n3. Результаты аудита и фильтрация (Порог: Score >= 6):")
    surviving_features = []

    for item in audit_results:
        status = "✅ ПРИНЯТО" if item['score'] >= 6 else "❌ ОТКЛОНЕНО"
        print(f"{status} | {item['feature_name']} (Score: {item['score']}) - {item['reasoning']}")
        
        if item['score'] >= 6:
            surviving_features.append(item['feature_name'])

    output_path = "data/processed/alpha_r1_features.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(surviving_features, f, indent=4)
        
    print(f"\nАУДИТ ЗАВЕРШЕН. Выжило фичей: {len(surviving_features)} из {len(features)}.")
    print(f"Элитный список сохранен в {output_path}")

if __name__ == "__main__":
    run_local_alpha_audit()