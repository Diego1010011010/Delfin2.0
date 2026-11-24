from influxdb_client import InfluxDBClient

url = "http://10.119.190.215:8086"
token = "QTmddP-GbRUtKUJUPEXG0Vkxein2M1311DVFWASKFVOh9VqoS7Eb9C5oF9O9ebPtNMcRss0nIL86E1XsExaVVQ=="
org = "union_station"
bucket = "meteorological_data"

try:
    client = InfluxDBClient(url=url, token=token, org=org)
    health = client.health()
    print("Health:", health)
    query_api = client.query_api()
    query = f'from(bucket:"{bucket}") |> range(start: -1h) |> limit(n:1)'
    result = query_api.query(query)
    print("Consulta exitosa, datos:", result)
except Exception as e:
    print("Error:", e)