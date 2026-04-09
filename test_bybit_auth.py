import os
import time
import hmac
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BYBIT_KEY")
API_SECRET = os.getenv("BYBIT_SECRET")

def test_auth():
    # 1. Данные для запроса
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    params = "accountType=UNIFIED" # Строка, которую мы подписываем
    
    # 2. Формируем "сырую" строку для подписи (как в доках V5)
    # Формула: timestamp + api_key + recv_window + query_string
    signature_raw = timestamp + API_KEY + recv_window + params
    
    # 3. Создаем саму подпись (печатаем её)
    signature = hmac.new(
        bytes(API_SECRET, "utf-8"), 
        signature_raw.encode("utf-8"), 
        hashlib.sha256
    ).hexdigest()

    print(f"--- DEBUG INFO ---")
    print(f"Timestamp: {timestamp}")
    print(f"Raw String to Sign: {signature_raw}")
    print(f"Generated Signature: {signature}")
    print(f"------------------")

    headers = {
        "X-BIP-API-KEY": API_KEY,
        "X-BIP-API-SIGN": signature,
        "X-BIP-API-TIMESTAMP": timestamp,
        "X-BIP-API-RECV-WINDOW": recv_window
    }

    url = f"https://api-testnet.bybit.com/v5/account/wallet-balance?{params}"
    res = requests.get(url, headers=headers)
    
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.text}")

test_auth()