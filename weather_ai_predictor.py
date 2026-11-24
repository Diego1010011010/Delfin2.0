#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
import joblib
import logging
import os
import json

# ===== CONFIGURACIÓN =====
INFLUXDB_URL = "http://10.186.33.215:8086"
INFLUXDB_TOKEN = "QTmddP-GbRUtKUJUPEXG0Vkxein2M1311DVFWASKFVOh9VqoS7Eb9C5oF9O9ebPtNMcRss0nIL86E1XsExaVVQ=="
INFLUXDB_ORG = "union_station"
INFLUXDB_BUCKET = "meteorological_data"

MODEL_TEMP_PATH = "model_temperatura.h5"
MODEL_RAIN_PATH = "model_lluvia.h5"
SCALER_PATH = "scaler_weather.pkl"
HISTORY_PATH = "predicciones_historial.json"

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeatherPredictorLocal:
    def __init__(self):
        self.influx_client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        self.scaler = None
        self.model_temp = None
        self.model_rain = None
        self.seq_length = 7  # Días de historia para predecir
        self.features = ['temperatura', 'presion', 'humedad', 'luz', 'calidad_aire', 'lluvia_intensidad']

    def fetch_local_data(self, days=90):
        """Obtiene datos de tu estación desde InfluxDB"""
        logger.info(f"Obteniendo datos locales de los últimos {days} días...")
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
            |> range(start: -{days}d)
            |> filter(fn: (r) => r._measurement == "weather_station")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        try:
            result = self.influx_client.query_api().query_data_frame(query)
            if result is None or result.empty:
                logger.error("No se encontraron datos en InfluxDB")
                return None
            df = result.copy()
            df['_time'] = pd.to_datetime(df['_time'])
            df = df.set_index('_time')
            df = df.sort_index()
            df = df.dropna()
            df['dayofyear'] = df.index.dayofyear
            df['month'] = df.index.month
            df['dayofweek'] = df.index.dayofweek
            df['tmax_tomorrow'] = df['temperatura'].shift(-1)
            df['rain_tomorrow'] = (df['lluvia_intensidad'].shift(-1) > 100).astype(int)
            df = df.dropna()
            logger.info(f"Datos cargados: {len(df)} registros")
            logger.info(f"Periodo: {df.index.min()} a {df.index.max()}")
            return df
        except Exception as e:
            logger.error(f"Error al obtener datos: {e}")
            return None

    def prepare_sequences(self, data, labels, seq_length=7):
        X_seq, y_seq = [], []
        for i in range(len(data) - seq_length):
            X_seq.append(data[i:i+seq_length])
            y_seq.append(labels[i+seq_length])
        return np.array(X_seq), np.array(y_seq)

    def train_models(self, days=90):
        """Entrena ambos modelos (temperatura y lluvia)"""
        df = self.fetch_local_data(days)
        if df is None or len(df) < 30:
            logger.error("No hay suficientes datos para entrenar")
            return False
        X = df[self.features + ['dayofyear', 'month', 'dayofweek']].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_seq, y_temp = self.prepare_sequences(X_scaled, df['tmax_tomorrow'].values, self.seq_length)
        _, y_rain = self.prepare_sequences(X_scaled, df['rain_tomorrow'].values, self.seq_length)
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_temp_train, y_temp_test = y_temp[:split_idx], y_temp[split_idx:]
        y_rain_train, y_rain_test = y_rain[:split_idx], y_rain[split_idx:]
        logger.info(f"Entrenamiento: {X_train.shape[0]} muestras")
        logger.info(f"Prueba: {X_test.shape[0]} muestras")

        # Modelo de temperatura
        logger.info("Entrenando modelo de temperatura...")
        self.model_temp = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(self.seq_length, X_train.shape[2])),
            Dropout(0.3),
            LSTM(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        self.model_temp.compile(optimizer='adam', loss='mse', metrics=['mae'])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model_temp.fit(X_train, y_temp_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)
        loss_temp, mae_temp = self.model_temp.evaluate(X_test, y_temp_test, verbose=0)
        logger.info(f" Error Absoluto Medio Temperatura: {mae_temp:.2f}°C")

        # Modelo de lluvia
        logger.info("Entrenando modelo de lluvia...")
        self.model_rain = Sequential([
            LSTM(32, activation='relu', input_shape=(self.seq_length, X_train.shape[2])),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model_rain.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])
        self.model_rain.fit(X_train, y_rain_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)
        y_rain_pred = (self.model_rain.predict(X_test) > 0.5).astype(int)
        accuracy_rain = accuracy_score(y_rain_test, y_rain_pred)
        logger.info(f" Precisión Lluvia: {accuracy_rain*100:.1f}%")

        self.save_models()
        return True

    def save_models(self):
        if self.model_temp:
            self.model_temp.save(MODEL_TEMP_PATH)
            logger.info(f"Modelo de temperatura guardado: {MODEL_TEMP_PATH}")
        if self.model_rain:
            self.model_rain.save(MODEL_RAIN_PATH)
            logger.info(f"Modelo de lluvia guardado: {MODEL_RAIN_PATH}")
        if self.scaler:
            joblib.dump(self.scaler, SCALER_PATH)
            logger.info(f"Scaler guardado: {SCALER_PATH}")

    def load_models(self):
        """Carga los modelos entrenados y recompila para predicción"""
        try:
            if os.path.exists(MODEL_TEMP_PATH):
                # Cargar sin compilar para evitar problemas con métricas
                self.model_temp = load_model(MODEL_TEMP_PATH, compile=False)
                logger.info("Modelo de temperatura cargado (sin compilar)")
                # Recompilar para uso en predicción (opcional, pero evita warnings)
                self.model_temp.compile(optimizer='adam', loss='mse', metrics=['mae'])

            if os.path.exists(MODEL_RAIN_PATH):
                self.model_rain = load_model(MODEL_RAIN_PATH, compile=False)
                logger.info("Modelo de lluvia cargado (sin compilar)")
                self.model_rain.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])

            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("Scaler cargado")

            return self.model_temp is not None and self.model_rain is not None
        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")
            return False

    def predict_future(self, days_ahead=7):
        if not self.load_models():
            logger.error("No hay modelos disponibles. Entrena primero.")
            return None
        df = self.fetch_local_data(days=30)
        if df is None or len(df) < self.seq_length:
            logger.error("No hay suficientes datos recientes")
            return None
        X_recent = df[self.features + ['dayofyear', 'month', 'dayofweek']].values[-self.seq_length:]
        X_recent_scaled = self.scaler.transform(X_recent)
        predictions = []
        current_seq = X_recent_scaled.copy()
        for day in range(days_ahead):
            X_pred = current_seq.reshape(1, self.seq_length, -1)
            temp_pred = self.model_temp.predict(X_pred, verbose=0)[0][0]
            rain_prob = self.model_rain.predict(X_pred, verbose=0)[0][0]
            future_date = datetime.now() + timedelta(days=day+1)
            predictions.append({
                'fecha': future_date.strftime('%Y-%m-%d'),
                'temperatura': round(temp_pred, 2),
                'probabilidad_lluvia': round(rain_prob * 100, 1),
                'llovera': rain_prob > 0.5
            })
            new_step = np.array([
                temp_pred, 1013.0, 65.0, 500.0, 300.0, 0.0,
                future_date.timetuple().tm_yday,
                future_date.month,
                future_date.weekday()
            ]).reshape(1, -1)
            new_step_scaled = self.scaler.transform(new_step)
            current_seq = np.roll(current_seq, -1, axis=0)
            current_seq[-1] = new_step_scaled
        return predictions

    def get_extreme_days(self, days_ahead=7):
        predictions = self.predict_future(days_ahead)
        if not predictions:
            return None
        hottest = max(predictions, key=lambda x: x['temperatura'])
        coldest = min(predictions, key=lambda x: x['temperatura'])
        return {
            'dia_mas_caluroso': {'fecha': hottest['fecha'], 'temperatura': hottest['temperatura']},
            'dia_mas_frio': {'fecha': coldest['fecha'], 'temperatura': coldest['temperatura']},
            'dias_con_lluvia': [p for p in predictions if p['llovera']],
            'prediccion_completa': predictions
        }

    def generate_report(self, days_ahead=7):
        predictions = self.predict_future(days_ahead)
        if not predictions:
            return None
        extremos = self.get_extreme_days(days_ahead)
        report = {
            'fecha_generacion': datetime.now().isoformat(),
            'datos_usados': 'Estación Union Station (datos locales)',
            'periodo': f'Próximos {days_ahead} días',
            'predicciones': predictions,
            'extremos': extremos
        }
        return report

def main():
    """Función principal"""
    predictor = WeatherPredictorLocal()

    # Verificar si hay modelos guardados
    if not os.path.exists(MODEL_TEMP_PATH) or not os.path.exists(MODEL_RAIN_PATH):
        logger.info("No se encontraron modelos. Entrenando con datos locales...")
        success = predictor.train_models(days=90)
        if not success:
            logger.error("Error en el entrenamiento")
            return
    else:
        logger.info("Modelos cargados desde disco")

    # Generar predicción
    logger.info("Generando predicción del clima...")
    report = predictor.generate_report(days_ahead=7)

    if report:
        print("\n" + "="*70)
        print("  PREDICCIÓN CLIMÁTICA - UNION STATION")
        print("="*70)
        print(f" Generado: {report['fecha_generacion']}")
        print(f" Datos: {report['datos_usados']}")
        print(f" Periodo: {report['periodo']}")

        print("\n PREDICCIONES DÍA POR DÍA:")
        print("-" * 70)

        for pred in report['predicciones']:
            status = []
            if pred['probabilidad_lluvia'] > 60:
                status.append(" ALTA PROB. LLUVIA")
            elif pred['probabilidad_lluvia'] > 30:
                status.append(" POSIBLE LLUVIA")
            else:
                status.append(" SECO")

            if pred['temperatura'] > 30:
                status.append(" CALUROSO")
            elif pred['temperatura'] < 15:
                status.append("❄️ FRÍO")

            print(f"\n {pred['fecha']}:")
            print(f"   Temperatura: {pred['temperatura']}°C")
            print(f"    Prob. lluvia: {pred['probabilidad_lluvia']}%")
            print(f"    {' | '.join(status)}")

        print("\n" + "="*70)
        print(" EXTREMOS:")
        print("-" * 70)
        if report['extremos']:
            print(f" Día más caluroso: {report['extremos']['dia_mas_caluroso']['fecha']} "
                  f"({report['extremos']['dia_mas_caluroso']['temperatura']}°C)")
            print(f"  Día más frío: {report['extremos']['dia_mas_frio']['fecha']} "
                  f"({report['extremos']['dia_mas_frio']['temperatura']}°C)")

            if report['extremos']['dias_con_lluvia']:
                print(f"\n Días con lluvia prevista:")
                for d in report['extremos']['dias_con_lluvia']:
                    print(f"   • {d['fecha']} ({d['probabilidad_lluvia']}%)")

        # Guardar en historial
        try:
            with open(HISTORY_PATH, 'a') as f:
                f.write(json.dumps(report, default=str) + '\n')
            print(f"\n✅ Predicción guardada en: {HISTORY_PATH}")
        except Exception as e:
            logger.warning(f"No se pudo guardar historial: {e}")

        print("\n" + "="*70)
    else:
        logger.error("No se pudo generar la predicción")

if __name__ == "__main__":
    main()