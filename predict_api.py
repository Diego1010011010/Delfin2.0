"""
API para Node-RED - Predicciones climáticas con IA
Actualización automática cada hora usando un hilo en segundo plano
"""
from datetime import datetime
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from weather_multi_output import WeatherPredictorLocal
import logging
import os
import json
import numpy as np
import threading
import time

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Iniciar el predictor
predictor = WeatherPredictorLocal()

# Cargar modelos al inicio
if not predictor.load_models():
    logger.warning("No se pudieron cargar los modelos. Entrena primero con weather_multi_output.py")
else:
    logger.info("Modelos cargados correctamente")

# Variables de cache
cached_report = None
last_update = None
_update_lock = threading.Lock()


def convert_to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serializable")


def update_predictions():
    global cached_report, last_update
    with _update_lock:
        try:
            logger.info("Actualizando predicción con datos recientes...")
            report = predictor.generate_report(days_ahead=7)
            if report:
                cached_report = report
                last_update = datetime.now()
                logger.info(f"Predicción actualizada correctamente a las {last_update}")
            else:
                logger.error("Falló la generación del reporte")
        except Exception as e:
            logger.error(f"Error al actualizar predicción: {e}")


def scheduler_loop():
    while True:
        time.sleep(3600)          # espera 1 hora
        update_predictions()


def start_background():
    """Primera carga sincrónica + hilo cada hora."""
    update_predictions()
    threading.Thread(target=scheduler_loop, daemon=True).start()
    logger.info("Hilo de actualización automática iniciado (cada 1 hora)")


# ------------------------------------------------------------------
# RUTA RAÍZ (esto es lo que faltaba y provocaba el "Not Found")
# ------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'mensaje': 'API de predicciones climáticas funcionando',
        'endpoints': {
            'GET  /api/predict':   'Predicción completa de 7 días',
            'POST /api/refresh':   'Forzar actualización',
            'GET  /api/extremes':  'Días extremos (opcional ?days=7)',
            'GET  /api/status':    'Estado del servicio'
        },
        'ultima_actualizacion': last_update.isoformat() if last_update else None
    })


@app.route('/api/predict', methods=['GET'])
def predict():
    global cached_report, last_update
    if cached_report is None:
        return jsonify({
            'success': False,
            'error': 'Aún no hay predicción disponible.'
        }), 503
    return Response(
        response=json.dumps({
            'success': True,
            'data': cached_report,
            'ultima_actualizacion': last_update.isoformat() if last_update else None
        }, default=convert_to_serializable),
        status=200,
        mimetype='application/json'
    )


@app.route('/api/refresh', methods=['POST'])
def refresh():
    update_predictions()
    return jsonify({
        'success': True,
        'message': 'Actualización forzada ejecutada',
        'ultima_actualizacion': last_update.isoformat() if last_update else None
    })


@app.route('/api/extremes', methods=['GET'])
def extremes():
    if cached_report and 'extremos' in cached_report:
        return Response(
            response=json.dumps({'success': True, 'data': cached_report['extremos']},
                                default=convert_to_serializable),
            status=200,
            mimetype='application/json'
        )
    try:
        days = request.args.get('days', 7, type=int)
        extremos = predictor.get_extreme_days(days)
        if extremos:
            return Response(
                response=json.dumps({'success': True, 'data': extremos},
                                    default=convert_to_serializable),
                status=200,
                mimetype='application/json'
            )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': False, 'error': 'No hay datos'}), 404


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'online',
        'modelos_cargados': getattr(predictor, 'model', None) is not None,
        'ultima_actualizacion': last_update.isoformat() if last_update else None,
        'cache_disponible': cached_report is not None
    })


if __name__ == '__main__':
    start_background()
    app.run(host='0.0.0.0', port=5000, debug=False)