# File: strategies/library/factory.py
import importlib
import re

class StrategyFactory:
    @staticmethod
    def load_from_config(config: dict):
        class_name = config.get("metadata", {}).get("strategy_class")
        if not class_name:
            raise ValueError("Config missing 'metadata.strategy_class'")

        # Convert CamelCase (GenericStrategy) to snake_case (generic_strategy)
        # This regex is much safer!
        file_name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        
        try:
            # Dynamically import the file from the library
            module = importlib.import_module(f"strategies.library.{file_name}")
            strategy_class = getattr(module, class_name)
            return strategy_class(config)
        except (ImportError, AttributeError) as e:
            print(f"❌ Strategy Factory Error: Could not find class '{class_name}' in 'strategies/library/{file_name}.py'")
            raise e