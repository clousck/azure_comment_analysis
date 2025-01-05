from flask import Flask, jsonify, request
from library.use_model import preprocesar_texto, predecir_sentimiento
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Ruta de prueba
@app.route('/test', methods=['GET'])
def hello_world():
    texto = "Me encantó este producto, es excelente"
    preprocesado = preprocesar_texto(texto)
    print(preprocesado)
    sentimiento = predecir_sentimiento(texto)
    print(sentimiento)

    return jsonify({'message': sentimiento})

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    # Obtener el texto desde el formulario
    texto = request.form.get('comentario', '')

    # Llamar a la función para predecir el sentimiento
    prediction = predecir_sentimiento(texto)
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
