import tensorflow as tf
import numpy as np

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("library/analisis_sentimientos.keras")

def predecir_sentimiento(texto):
    preprocesado = tf.constant([texto])

    prediccion = modelo.predict(preprocesado)[0][0]

    if prediccion > 0:
        return 1
    else:
        return 0



# # Texto de entrada
# sample_text = (
#     '''I love it.'''
# )

# # Realizar predicciones con el modelo
# predictions = modelo.predict(tf.constant([sample_text]))

# # Mostrar resultados
# print(f"Predicciones: {predictions[0][0]}")