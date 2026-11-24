#!/usr/bin/env python3
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import joblib
import logging
import os
import json

INFLUXDB_URL = "http://10.186.33.215:8086"
INFLUXDB_TOKEN = "QTmddP-GbRUtKUJUPEXG0Vkxein2M1311DVFWASKFVOh9VqoS7Eb9C5oF9O9ebPtNMcRss0nIL86E1XsExaVVQ=="
INFLUXDB_ORG = "union_station"
INFLUXDB_BUCKET = "meteorological_data"

MODEL_MULTI_PATH = "model_multi_output.h5"
SCALER_FEATURES_PATH = "scaler_features.pkl"
SCALER_TARGET_TEMP_PATH = "scaler_target_temp.pkl"
HISTORY_PATH = "predicciones_historial.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeatherPredictorLocal:
    def __init__(self):
        self.influx_client = InfluxDBClient(
            url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG
        )
        self.seq_length = 24          # CAMBIO: antes 168
        self.forecast_horizon = 7
        self.features = ['temperatura', 'presion', 'humedad', 'luz',
                         'calidad_aire', 'lluvia_intensidad']
        self.scaler = None
        self.scaler_target_temp = None
        self.model = None
        self.mean_rain_intensity = None

    def fetch_local_data(self, days=7):          # CAMBIO: antes 365
        logger.info(f"Obteniendo datos de los últimos {days} días...")
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
            |> range(start: -{days}d)
            |> filter(fn: (r) => r._measurement == "weather_station")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        try:
            result = self.influx_client.query_api().query_data_frame(query)
            if result is None or result.empty:
                logger.error("No se encontraron datos")
                return None
            df = result.copy()
            df['_time'] = pd.to_datetime(df['_time'])
            df = df.set_index('_time').sort_index()

            available_cols = [col for col in self.features if col in df.columns]
            if not available_cols:
                logger.error("Ninguna de las características requeridas está presente")
                return None
            df = df[available_cols]

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.resample("1min").mean()
            df = df.interpolate(method="time")
            df = df.dropna()

            if 'temperatura' in df:
                df = df[(df["temperatura"] >= 10) & (df["temperatura"] <= 50)]
            if 'presion' in df:
                df = df[(df["presion"] >= 850) & (df["presion"] <= 1100)]
            if 'humedad' in df:
                df = df[(df["humedad"] >= 0) & (df["humedad"] <= 100)]

            # CAMBIO: neutralizar sensor de lluvia viejo (roto)
            if 'lluvia_intensidad' in df:
                df['lluvia_intensidad'] = 0.0

            logger.info(f"Datos cargados: {len(df)} registros (minutos)")
            df = df.resample("1h").mean()
            df = df.dropna()
            logger.info(f"Datos horarios: {len(df)} registros")
            return df
        except Exception as e:
            logger.error(f"Error: {e}")
            return None

    def _add_cyclical_features(self, df):
        df['sin_dayofyear'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['cos_dayofyear'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        df['sin_month'] = np.sin(2 * np.pi * df.index.month / 12)
        df['cos_month'] = np.cos(2 * np.pi * df.index.month / 12)
        df['sin_dayofweek'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['cos_dayofweek'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['sin_hour'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df.index.hour / 24)
        return df

    # CAMBIO: ventanas en HORAS (antes usaba 180 = 180h, dejaba todo NaN)
    def _add_rolling_features(self, df):
        df['temp_ma6']   = df['temperatura'].rolling(6).mean()
        df['temp_ma12']  = df['temperatura'].rolling(12).mean()
        df['temp_ma24']  = df['temperatura'].rolling(24).mean()
        df['rain_ma6']   = df['lluvia_intensidad'].rolling(6).mean()
        df['rain_ma12']  = df['lluvia_intensidad'].rolling(12).mean()
        df['temp_trend'] = df['temperatura'].diff().rolling(3).mean()
        df['rain_trend'] = df['lluvia_intensidad'].diff().rolling(3).mean()
        df['temp_1h']    = df['temperatura'].shift(1)
        df['temp_3h']    = df['temperatura'].shift(3)
        df['temp_6h']    = df['temperatura'].shift(6)
        df['temp_5min']  = df['temperatura'].rolling(3).mean()
        df['temp_15min'] = df['temperatura'].rolling(6).mean()
        df['temp_60min'] = df['temperatura'].rolling(12).mean()
        return df

    def prepare_multi_output_data(self, df):
        df = self._add_cyclical_features(df)
        df = self._add_rolling_features(df)
        df = df.dropna()

        feature_cols = self.features + [
            'sin_dayofyear', 'cos_dayofyear', 'sin_month', 'cos_month',
            'sin_dayofweek', 'cos_dayofweek', 'sin_hour', 'cos_hour',
            'temp_ma6', 'temp_ma12', 'temp_ma24', 'rain_ma6', 'rain_ma12',
            'temp_trend', 'temp_5min', 'temp_15min', 'temp_60min', 'rain_trend'
        ]
        X_raw = df[feature_cols].values

        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        X_seq, Y_temp, Y_rain = [], [], []
        for i in range(len(df) - self.seq_length - self.forecast_horizon + 1):
            X_seq.append(X_scaled[i:i+self.seq_length])
            Y_temp.append(df['temperatura'].values[i+self.seq_length : i+self.seq_length+self.forecast_horizon])
            Y_rain.append((df['lluvia_intensidad'].values[i+self.seq_length : i+self.seq_length+self.forecast_horizon] > 100).astype(int))
        return np.array(X_seq), np.array(Y_temp), np.array(Y_rain)

    def build_multi_output_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = LSTM(128, activation='tanh', return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        temp_out = Dense(32, activation='relu')(x)
        temp_out = Dense(self.forecast_horizon, activation='linear', name='temperature_output')(temp_out)
        rain_out = Dense(self.forecast_horizon, activation='sigmoid', name='rain_output')(x)
        model = Model(inputs=inputs, outputs=[temp_out, rain_out])
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss={'temperature_output': MeanSquaredError(), 'rain_output': BinaryCrossentropy()},
            loss_weights={'temperature_output': 1.0, 'rain_output': 1.0},
            metrics={'temperature_output': 'mae', 'rain_output': 'accuracy'}
        )
        return model

    def train_models(self, days=7):              # CAMBIO: antes 365
        df = self.fetch_local_data(days)
        if df is None or len(df) < 100:
            logger.error("Datos insuficientes")
            return False

        rainy = df[df['lluvia_intensidad'] > 100]
        self.mean_rain_intensity = rainy['lluvia_intensidad'].mean() if len(rainy) > 0 else 150.0
        logger.info(f"Intensidad media de lluvia: {self.mean_rain_intensity:.1f}")

        X_seq, Y_temp, Y_rain = self.prepare_multi_output_data(df)

        split = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split], X_seq[split:]
        Y_temp_train, Y_temp_test = Y_temp[:split], Y_temp[split:]
        Y_rain_train, Y_rain_test = Y_rain[:split], Y_rain[split:]

        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

        self.scaler_target_temp = MinMaxScaler()
        Y_temp_scaled_train = self.scaler_target_temp.fit_transform(Y_temp_train)
        Y_temp_scaled_test = self.scaler_target_temp.transform(Y_temp_test)

        model = self.build_multi_output_model((self.seq_length, X_train.shape[2]))
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)

        history = model.fit(
            X_train,
            [Y_temp_scaled_train, Y_rain_train],
            validation_split=0.2,
            epochs=100,
            batch_size=16,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        loss, loss_temp, loss_rain, mae_temp, acc_rain = model.evaluate(
            X_test, [Y_temp_scaled_test, Y_rain_test], verbose=0)
        mae_temp_descaled = mae_temp * self.scaler_target_temp.scale_[0]
        logger.info(f"MAE Temperatura (desescalado): {mae_temp_descaled:.2f}°C")
        logger.info(f"Accuracy Lluvia: {acc_rain*100:.1f}%")

        self.model = model
        self.save_models()
        return True

    def save_models(self):
        if self.model:
            self.model.save(MODEL_MULTI_PATH)
            logger.info(f"Modelo guardado: {MODEL_MULTI_PATH}")
        if self.scaler:
            joblib.dump(self.scaler, SCALER_FEATURES_PATH)
            logger.info("Scaler features guardado")
        if self.scaler_target_temp:
            joblib.dump(self.scaler_target_temp, SCALER_TARGET_TEMP_PATH)
            logger.info("Scaler target temp guardado")

    def load_models(self):
        try:
            if os.path.exists(MODEL_MULTI_PATH):
                self.model = load_model(MODEL_MULTI_PATH, compile=False)
                self.model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss={'temperature_output': 'mse', 'rain_output': 'binary_crossentropy'},
                    metrics={'temperature_output': 'mae', 'rain_output': 'accuracy'}
                )
                logger.info(" Modelo multi-salida cargado")
            else:
                logger.warning(" No se encontró modelo multi-salida")
                return False

            if os.path.exists(SCALER_FEATURES_PATH):
                self.scaler = joblib.load(SCALER_FEATURES_PATH)
                logger.info(" Scaler de características cargado")
            else:
                logger.warning(" No se encontró scaler de características")
                return False

            if os.path.exists(SCALER_TARGET_TEMP_PATH):
                self.scaler_target_temp = joblib.load(SCALER_TARGET_TEMP_PATH)
                logger.info(" Scaler de target temperatura cargado")
            else:
                logger.warning(" No se encontró scaler de target temperatura")
                return False

            return True
        except Exception as e:
            logger.error(f" Error cargando modelos: {e}")
            return False

    def predict_future(self, days_ahead=7):
        if days_ahead != self.forecast_horizon:
            logger.warning(f"El modelo fue entrenado para {self.forecast_horizon} días.")
            if days_ahead > self.forecast_horizon:
                days_ahead = self.forecast_horizon

        if not self.load_models():
            logger.error("No hay modelos cargados")
            return None

        df = self.fetch_local_data(days=7)          # CAMBIO: antes 30
        if df is None or len(df) < self.seq_length:
            logger.error("Datos recientes insuficientes")
            return None

        df = self._add_cyclical_features(df)
        df = self._add_rolling_features(df)
        df = df.dropna()

        feature_cols = self.features + [
            'sin_dayofyear', 'cos_dayofyear', 'sin_month', 'cos_month',
            'sin_dayofweek', 'cos_dayofweek', 'sin_hour', 'cos_hour',
            'temp_ma6', 'temp_ma12', 'temp_ma24', 'rain_ma6', 'rain_ma12',
            'temp_trend', 'temp_5min', 'temp_15min', 'temp_60min', 'rain_trend'
        ]
        X_recent = df[feature_cols].values[-self.seq_length:]
        X_recent_scaled = self.scaler.transform(X_recent).reshape(1, self.seq_length, -1)

        temp_scaled, rain_prob = self.model.predict(X_recent_scaled, verbose=0)
        temp_pred = self.scaler_target_temp.inverse_transform(temp_scaled)[0]
        rain_pred = (rain_prob > 0.5).astype(int)[0]

        last_date = df.index[-1]
        predictions = []
        for i in range(days_ahead):
            future_date = last_date + timedelta(days=i+1)
            predictions.append({
                'fecha': future_date.strftime('%Y-%m-%d'),
                'temperatura': round(float(temp_pred[i]), 2),
                'probabilidad_lluvia': round(float(rain_prob[0][i]) * 100, 1),
                'llovera': bool(rain_pred[i])
            })
        return predictions

    def get_extreme_days(self, days_ahead=7):
        preds = self.predict_future(days_ahead)
        if not preds:
            return None
        return {
            'dia_mas_caluroso': max(preds, key=lambda x: x['temperatura']),
            'dia_mas_frio': min(preds, key=lambda x: x['temperatura']),
            'dias_con_lluvia': [p for p in preds if p['llovera']],
            'prediccion_completa': preds
        }

    def generate_report(self, days_ahead=7):
        preds = self.predict_future(days_ahead)
        if not preds:
            return None
        extremos = self.get_extreme_days(days_ahead)
        return {
            'fecha_generacion': datetime.now().isoformat(),
            'datos_usados': 'Estación Union Station',
            'periodo': f'Próximos {days_ahead} días',
            'predicciones': preds,
            'extremos': extremos
        }


def main():
    predictor = WeatherPredictorLocal()
    if not os.path.exists(MODEL_MULTI_PATH):
        logger.info("Entrenando modelo multi-salida...")
        if not predictor.train_models(days=7):      # CAMBIO: antes 365
            logger.error("Entrenamiento fallido")
            return
    else:
        logger.info("Cargando modelo existente")
        predictor.load_models()

    report = predictor.generate_report(days_ahead=7)
    if report:
        print(json.dumps(report, indent=2, default=str))
    else:
        logger.error("No se pudo generar reporte")


if __name__ == "__main__":
    main()