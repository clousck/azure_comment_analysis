import tensorflow as tf
import numpy as np

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("library/analisis_sentimientos.keras")

# Función para preprocesar el comentario
def preprocesar_texto(texto, max_length=100):
    """
    Preprocesa el texto para que sea compatible con el modelo.
    Ajusta esta función según cómo hayas procesado los datos al entrenar el modelo.
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Tokenizar el texto (reemplaza con el tokenizer usado durante el entrenamiento)
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts([texto])  # Puedes cargar el tokenizer entrenado aquí si lo tienes guardado

    # Convertir el texto a secuencias de enteros
    secuencias = tokenizer.texts_to_sequences([texto])
    
    # Asegúrate de que la longitud de las secuencias sea uniforme
    secuencias_padded = pad_sequences(secuencias, maxlen=max_length, padding="post", truncating="post")

    return secuencias_padded

# Función para predecir el sentimiento
def predecir_sentimiento(texto):
    """
    Recibe un comentario de texto, lo procesa y devuelve la predicción del modelo.
    """
    texto_procesado = preprocesar_texto(texto)
    prediccion = modelo.predict(texto_procesado)

    # Interpretar el resultado (asume que el modelo devuelve un valor entre 0 y 1)
    if prediccion[0] > 0.5:
        return 1
    else:
        return 0

def predict(texto):
    prediccion = modelo.predict(texto)
    return prediccion
    
    

# Ejemplo de uso
comentario = "Me encantó este producto, es excelente"
resultado = predecir_sentimiento(comentario)
print(f"El comentario '{comentario}' es de sentimiento: {resultado}")