from weather_ai_predictor import WeatherPredictorLocal 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== INICIANDO ENTRENAMIENTO ===")
    logger.info("Usando datos de la estación Union Station")
    
    predictor = WeatherPredictorLocal()
    
    # Entrenar con todos los datos disponibles
    success = predictor.train_models(days=90)  # Últimos 90 días
    
    if success:
        logger.info("✅ Entrenamiento completado exitosamente")
        logger.info(f"📁 Modelos guardados:")
        logger.info(f"   - {__import__('os').path.abspath('model_temperatura.h5')}")
        logger.info(f"   - {__import__('os').path.abspath('model_lluvia.h5')}")
        logger.info(f"   - {__import__('os').path.abspath('scaler_weather.pkl')}")
    else:
        logger.error("❌ Falló el entrenamiento")

if __name__ == "__main__":
    main()