import pandas as pd
from typing import Dict

class ExecutionEngine:
    def __init__(self, broker):
        self.broker = broker

    def execute(self, order_payload: dict):
        if order_payload and order_payload.get('size', 0) > 0:
            self.broker.open_trade(order_payload)