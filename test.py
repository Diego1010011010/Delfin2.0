import pandas as pd #Para trabajar con tablas.
import numpy as np #Para cálculos matemáticos.
from sklearn.model_selection import train_test_split #Para dividir datos aunque después realmente no se utiliza
from sklearn.preprocessing import StandardScaler #Normaliza los datos.
from sklearn.metrics import accuracy_score, mean_absolute_error #calcula Accuracy (% de aciertos) y el MAE
from tensorflow.keras.models import Sequential #Crea la red neuronal 
from tensorflow.keras.layers import Dense, Dropout, LSTM  #LSTM para series temporales importa eso y apaga neuronas aleatorias y hace la capa densa 
from tensorflow.keras.callbacks import EarlyStopping  #Detiene el entrenamiento cuando deja de mejorar.
from datetime import datetime
# 1. CARGAR DATOS CLIMÁTICOS (usando NOAA directo)
import requests #se conecta a internet 

def fetch_mazatlan_data(start_date, end_date):#Función que descargará datos climáticos con fecha de incio y fin 
    #cordenadas del area 
    LAT = 23.2494
    LON = -106.4111

    url = (                                              #la url 
        f"https://archive-api.open-meteo.com/v1/archive" #API de clima histórico.
        f"?latitude={LAT}"#latitud
        f"&longitude={LON}"#longitud
        f"&start_date={start_date}"#Fecha inicial
        f"&end_date={end_date}"#Fecha final
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum" #solicita la temperature_2m_max Temperatura máxima temperature_2m_min Temperatura mínima.precipitation_sumLluvia.
        f"&timezone=auto"#Usa zona horaria automática
    )

    response = requests.get(url)  #hce la petición a internet.
    data = response.json()         #Convierte la respuesta a diccionario Python.

    df = pd.DataFrame({            #construye una tabla 
        'DATE': data['daily']['time'], #fechas
        'TMAX': data['daily']['temperature_2m_max'],#temperatura maxima 
        'TMIN': data['daily']['temperature_2m_min'], #temperatura minima 
        'PRCP': data['daily']['precipitation_sum']  #cantidad de lluvia 
    })

    df['DATE'] = pd.to_datetime(df['DATE']) #Transforma texto en fecha real.

    return df  #devuelve el dataframe
fecha_hoy = datetime.today().strftime('%Y-%m-%d')

df = fetch_mazatlan_data(
    '2018-01-01',
     fecha_hoy         #descarga los datos por fechas 
)
print(f"Descargados {len(df)} días de datos")

# 2. FEATURE ENGINEERING TEMPORAL
# Crear características basadas en tiempo
df['dayofyear'] = df['DATE'].dt.dayofyear
df['month'] = df['DATE'].dt.month
df['dayofweek'] = df['DATE'].dt.dayofweek

# Crear target para clasificación (¿lloverá mañana?)
df['rain_tomorrow'] = (df['PRCP'].shift(-1) > 0).astype(int)  #Mueve la columna un día hacia arriba.

# Crear target para regresión (temperatura máxima mañana)
df['tmax_tomorrow'] = df['TMAX'].shift(-1) #Guardará la temperatura máxima del día siguiente y desplaza la columna un día.

# Eliminar filas con NaN (por el shift)
df = df.dropna() #El último día queda sin "mañana".


# 3. SELECCIONAR FEATURES
features = ['TMAX', 'TMIN', 'PRCP', 'dayofyear', 'month', 'dayofweek'] #Entradas de la red neuronal.
target_classification = 'rain_tomorrow'  # Para:va a llover
target_regression = 'tmax_tomorrow'      # Para: predecir temperatura máxima

X = df[features]                      #entradas
y_clasif = df[target_classification]   # Clasificación binaria si o no 
y_regres = df[target_regression]       # Regresión temperatura de mañana


# 4. ESCALADO (ya no necesitas OneHotEncoder ni imputer complejo)

scaler = StandardScaler() #crea normalizador
X_scaled = scaler.fit_transform(X) #normaliza los datos 


# 5. CREAR SECUENCIAS PARA LSTM (importante para series temporales)
def create_sequences(data, labels, seq_length=7): #funcion para crear sequencias 
    #Crea secuencias de 7 días para predecir el día 8
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(labels[i+seq_length])
    return np.array(X_seq), np.array(y_seq)
 #guarda los dias y se recorre de dias 
SEQ_LENGTH = 7  # Usar 7 días para predecir el día 8
X_seq, y_seq = create_sequences(X_scaled, y_regres, SEQ_LENGTH) #genera todas las sequencias 

# Dividir en train/test (manteniendo orden temporal)
split_idx = int(len(X_seq) * 0.8)        #80 de entrenamiento y 20 de prueba 
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

print(f"Forma de X_train: {X_train.shape}")  # (días, 7 días, 6 features)

# 6. RED NEURONAL CON LSTM (para series temporales)
#empieza el juice 
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(SEQ_LENGTH, X_train.shape[2])),
    Dropout(0.3),#64 nuronas con memoria y apaga 30%
    LSTM(32, activation='relu'),#funcion de activacion relu 32 neuronas y 16 ocultas 
    Dropout(0.3),           
    Dense(16, activation='relu'),
    Dense(1)  # Salida: temperatura predicha
])

model.compile(optimizer='adam', loss='mse', metrics=['mae']) #algoritmo de optimizacion con el error cuadrático medio

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar
print("Entrenando red LSTM para predicción de temperatura...") #aprende patrones climaticos 
history = model.fit(X_train, y_train, 
                    validation_split=0.2,
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1)

# Evaluar
loss, mae = model.evaluate(X_test, y_test, verbose=0) #Calcula qué tan bien predice.
print(f"\n Error Absoluto Medio en prueba: {mae:.2f} grados")


# 7. PREDECIR EL DÍA MÁS CALUROSO/HELADO

# Predecir todos los días del conjunto de prueba
y_pred = model.predict(X_test) #predice las temperaturas 

# Encontrar el día más caluroso predicho
hottest_idx = np.argmax(y_pred) #la mas alta 
coldest_idx = np.argmin(y_pred) #la mas baja 

print("\n" + "="*50)
print(" PREDICCIONES CLIMÁTICAS")
print("="*50)
print(f" Día más CALUROSO predicho: {y_pred[hottest_idx][0]:.1f}°C")
print(f" Día más HELADO predicho: {y_pred[coldest_idx][0]:.1f}°C")


# 8. CLASIFICACIÓN: predice si va a llover 
from tensorflow.keras.losses import BinaryCrossentropy #mide qué tan bien las predicciones del modelo coinciden con las etiquetas reales

# Para clasificación binaria (lluvia sí/no)
model_clasif = Sequential([
    LSTM(32, activation='relu', input_shape=(SEQ_LENGTH, X_train.shape[2])),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoide para probabilidad Devuelve una probabilidad entre 0 y 1
])

model_clasif.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])

# Crear secuencias para clasificación
y_clasif_seq = df[target_classification].values
_, y_clasif_seq_ready = create_sequences(X_scaled, y_clasif_seq, SEQ_LENGTH)
y_clasif_train, y_clasif_test = y_clasif_seq_ready[:split_idx], y_clasif_seq_ready[split_idx:]

# Entrenar
model_clasif.fit(X_train, y_clasif_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)

# Evaluar
y_clasif_pred = (model_clasif.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_clasif_test, y_clasif_pred)
print(f"\n Precisión para predecir si lloverá mañana: {accuracy*100:.1f}%")