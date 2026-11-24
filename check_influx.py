# check_influx.py
from influxdb_client import InfluxDBClient
import pandas as pd

INFLUXDB_URL    = "http://10.186.33.215:8086"
INFLUXDB_TOKEN  = "QTmddP-GbRUtKUJUPEXG0Vkxein2M1311DVFWASKFVOh9VqoS7Eb9C5oF9O9ebPtNMcRss0nIL86E1XsExaVVQ=="
INFLUXDB_ORG    = "union_station"
INFLUXDB_BUCKET = "meteorological_data"

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
api = client.query_api()

q1 = f'''
from(bucket: "{INFLUXDB_BUCKET}")
  |> range(start: -365d)
  |> filter(fn: (r) => r._measurement == "weather_station")
  |> filter(fn: (r) => r._field == "temperatura")
  |> keep(columns: ["_time"])
'''
df_time = api.query_data_frame(q1)
if isinstance(df_time, list):
    df_time = pd.concat(df_time, ignore_index=True)

if df_time is None or df_time.empty:
    print("NO hay datos de temperatura en los ultimos 365 dias.")
else:
    df_time["_time"] = pd.to_datetime(df_time["_time"], utc=True)
    print(f"Registros temperatura 365d: {len(df_time)}")
    print(f"Desde: {df_time['_time'].min()}")
    print(f"Hasta: {df_time['_time'].max()}")
    print(f"Rango: {(df_time['_time'].max() - df_time['_time'].min()).days} dias")

q2 = f'''
import "influxdata/influxdb/schema"
schema.fieldKeys(
  bucket: "{INFLUXDB_BUCKET}",
  predicate: (r) => r._measurement == "weather_station",
  start: -365d
)
'''
df_fields = api.query_data_frame(q2)
if isinstance(df_fields, list):
    df_fields = pd.concat(df_fields, ignore_index=True)

print("\nCampos disponibles en 'weather_station':")
if df_fields is not None and not df_fields.empty:
    print(df_fields[["_value"]].to_string(index=False))
else:
    print("  (ninguno)")

client.close()