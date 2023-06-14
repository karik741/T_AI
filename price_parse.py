import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'T_AI.settings')
django.setup()


from cabinet.models import BtcPrice
import time
import dateparser
import pytz
from binance.client import Client
from datetime import datetime

api_key = "your-api-key"
api_secret = "your-api-secret"

client = Client(api_key, api_secret)


# Функция для преобразования datetime объекта в метку времени Binance
def date_to_milliseconds(date_str):
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    d = dateparser.parse(date_str)
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    return int((d - epoch).total_seconds() * 1000)


# Функция для получения данных
def get_historical_klines(symbol, interval, start_str, end_str=None):
    output_data = []

    limit = 500
    timeframe = date_to_milliseconds(start_str)

    if end_str:
        end_time = date_to_milliseconds(end_str)
    else:
        end_time = int(round(time.time() * 1000))

    while True:
        try:
            temp_data = client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                startTime=timeframe,
                endTime=end_time
            )

            if not temp_data:
                break

            print(temp_data)

            output_data += temp_data

            timeframe = temp_data[len(temp_data) - 1][0] + 1

            time.sleep(0.15)

        except Exception as e:
            print(f"Caught exception: {e}")
            break

    return output_data


btc_data = get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 Jan, 2018")
btc_price = {}

for data in btc_data:
    timestamp = data[0] // 1000
    open_price = data[1]
    btc_price[timestamp] = open_price


model_data = []


for time, price in btc_price.items():
    print(f'{time}: {price}')
    model_data.append(BtcPrice(time=datetime.fromtimestamp(time), price=price))

BtcPrice.objects.all().delete()
chunk_size = 500
for i in range(0, len(model_data), chunk_size):
    BtcPrice.objects.bulk_create(model_data[i:i+chunk_size])
