import json
from pathlib import Path

# Legacy terms that must be standardized to lowercase Pandas columns
CASE_CORRECTIONS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
}

def fix_case_sensitivity(obj):
    """Recursively searches dictionaries and lists to fix uppercase price targets."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Fix keys if they happen to be uppercase price words
            new_key = CASE_CORRECTIONS.get(k, k)
            new_dict[new_key] = fix_case_sensitivity(v)
        return new_dict
    elif isinstance(obj, list):
        return [fix_case_sensitivity(item) for item in obj]
    elif isinstance(obj, str):
        # Fix values (e.g., {"source": "Close"} -> {"source": "close"})
        return CASE_CORRECTIONS.get(obj, obj)
    else:
        return obj

def process_configs(configs_dir: str = "strategies/configs/archive"):
    base_path = Path(configs_dir)
    if not base_path.exists():
        print(f"Directory not found: {configs_dir}")
        return

    updated_count = 0

    for json_file in base_path.rglob("*.json"):
        with open(json_file, 'r') as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON: {json_file.name}")
                continue

        needs_update = False

        # --- 1. Fix Strategy Class Mapping ---
        if "metadata" not in config:
            config["metadata"] = {}
            
        if "strategy_class" not in config["metadata"]:
            # Convert 'asian_box_breakout' -> 'AsianBoxBreakoutStrategy'
            base_name = json_file.stem
            camel_case_name = ''.join(word.title() for word in base_name.split('_'))
            strategy_class_name = f"{camel_case_name}Strategy"
            
            config["metadata"]["strategy_class"] = strategy_class_name
            needs_update = True

        # --- 2. Fix Case Sensitivity Drift ---
        fixed_config = fix_case_sensitivity(config)
        
        # Check if the case fixer actually changed anything
        if fixed_config != config:
            config = fixed_config
            needs_update = True

        # --- 3. Save if changes were made ---
        if needs_update:
            with open(json_file, 'w') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"✅ Fixed: {json_file.name} -> Class: {config['metadata'].get('strategy_class')}")
            updated_count += 1

    print(f"\n--- Process Complete: Fixed {updated_count} files ---")

if __name__ == "__main__":
    process_configs()