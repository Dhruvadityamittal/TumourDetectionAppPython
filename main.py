from flask import Flask, request
# import pandas as pd
from predict import predict
from PIL import Image
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

@app.route("/")
def hello():
    return  "Hello World!"

@app.route("/predict", methods=["POST"])
def predict_data():
    

    """
    Hello
    ---
    parameters:  
      - name: brain_scan_file
        in: formData
        type: file
        required: true

      - name: model_name
        in: formData
        type: string
        enum: ['ResNet', 'GoogleNet', 'MobileNet']
        required: true
        default : MobileNet
    
    responses:
        200:
            description: The output files
    """

    # model_name = request.args.get("model_name")
    model_name = request.form['model_name']
    print('Model Name is ', model_name)
    image = request.files.get("brain_scan_file")
    
    prediction = predict(model_name,image)
    
    
    return "Predicted Report :" + str(prediction)

if(__name__ == "__main__"):
    app.run(host='0.0.0.0',port=8000)