import os
import re
import json

def build_feature_inventory():
    features_dir = "shared/features"
    inventory = []
    
    print("1. Сканирование директории фичей...")
    if not os.path.exists(features_dir):
        print(f"❌ Ошибка: Папка {features_dir} не найдена.")
        return

    # Регулярное выражение для поиска @provides_features('feat1', 'feat2')
    pattern = re.compile(r"@provides_features\((.*?)\)")
    
    for filename in os.listdir(features_dir):
        if filename.endswith(".py") and filename not in ["__init__.py", "decorators.py"]:
            filepath = os.path.join(features_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                matches = pattern.findall(content)
                
                for match in matches:
                    # Очищаем строку от кавычек и пробелов, разбиваем по запятым
                    cleaned = match.replace("'", "").replace('"', "").replace(" ", "")
                    features = cleaned.split(',')
                    
                    for feat in features:
                        if feat: # Защита от пустых строк
                            inventory.append({
                                "feature_name": feat,
                                "module": filename
                            })
                            
    # Сортируем по имени модуля для красоты
    inventory.sort(key=lambda x: (x['module'], x['feature_name']))
    
    print(f"2. Инвентаризация завершена. Найдено {len(inventory)} уникальных фичей.")
    
    out_path = "data/processed/feature_registry.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=4)
        
    print(f"✅ Реестр успешно сохранен в {out_path}")
    print("\nПревью нескольких найденных фичей:")
    for item in inventory[:5]:
        print(f" 🔹 {item['feature_name']} (в {item['module']})")
    print(" ...")

if __name__ == "__main__":
    build_feature_inventory()