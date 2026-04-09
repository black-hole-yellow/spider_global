import os
import json
import time
import requests

# Настройки LLM для максимальной аналитической глубины (Gemma 4)
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma4:31b" # Измени на "gemma4:26b", если VRAM меньше 24GB
PASSING_SCORE = 6

def get_llm_evaluation(feature_name: str, module_name: str) -> dict:
    """Отправляет фичу на аудит в локальную LLM (Gemma 4) и требует строгий JSON."""
    
    prompt = f"""
    You are a Senior Quantitative Researcher at a top-tier hedge fund.
    We are building a machine learning model (Transformer) to predict the 15-minute timeframe of the GBP/USD forex pair.
    
    Evaluate the following feature for its microstructural, statistical, or theoretical edge.
    Feature Name: '{feature_name}' (located in module '{module_name}')
    
    Provide a score from 1 to 10. 
    1 = Pure noise, visually biased, or useless for ML.
    10 = Extremely robust, captures real institutional flow, liquidity, or chaos theory dynamics.
    
    You MUST respond with ONLY a valid JSON object and no other text.
    Format required:
    {{
        "feature": "{feature_name}",
        "score": <int>,
        "reasoning": "<1-2 sentences explaining the quantitative edge or lack thereof>"
    }}
    """
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json", # Форсируем вывод JSON
        "stream": False,
        "options": {
            "temperature": 0.0, # Строгая детерминированность, никакой отсебятины
            "num_predict": 256  # Ограничиваем длину ответа для ускорения
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        result_text = response.json().get("response", "{}")
        return json.loads(result_text)
        
    except Exception as e:
        print(f"\n⚠️ Ошибка вызова LLM для {feature_name}: {e}")
        return {"feature": feature_name, "score": 0, "reasoning": "LLM Parsing Failure or Timeout"}

def run_alpha_screener():
    registry_path = "data/processed/feature_registry.json"
    output_path = "data/processed/alpha_r1_features.json"
    audit_log_path = "data/processed/alpha_audit_log.json"
    
    print(f"1. Загрузка реестра фичей из {registry_path}...")
    if not os.path.exists(registry_path):
        print("❌ Реестр не найден. Сначала запусти build_feature_registry.py")
        return
        
    with open(registry_path, 'r', encoding='utf-8') as f:
        inventory = json.load(f)
        
    print(f"Найдено {len(inventory)} фичей для аудита. Запускаем ИИ-Кванта ({OLLAMA_MODEL})...")
    
    surviving_features = []
    audit_log = []
    
    for i, item in enumerate(inventory):
        feat = item["feature_name"]
        mod = item["module"]
        
        print(f"[{i+1}/{len(inventory)}] Анализ: {feat}...", end=" ", flush=True)
        
        evaluation = get_llm_evaluation(feat, mod)
        score = evaluation.get("score", 0)
        
        audit_log.append(evaluation)
        
        if score >= PASSING_SCORE:
            print(f"✅ Прошла (Score: {score})")
            surviving_features.append(feat)
        else:
            print(f"❌ Отклонена (Score: {score})")
            
        time.sleep(1.0) # Даем GPU немного остыть (1 секунда) между запросами
        
    print("\n2. Аудит завершен!")
    print(f"🛡️ Выжило фичей: {len(surviving_features)} из {len(inventory)} (Порог: {PASSING_SCORE}+)")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(surviving_features, f, indent=4)
        
    with open(audit_log_path, 'w', encoding='utf-8') as f:
        json.dump(audit_log, f, indent=4)
        
    print(f"✅ Белый список (Alpha-R1) сохранен в: {output_path}")
    print(f"✅ Полный лог обоснований сохранен в: {audit_log_path}")

if __name__ == "__main__":
    run_alpha_screener()