from flask import Flask, render_template, request, redirect, url_for
import requests
import json
from PIL import Image

app = Flask(__name__)

# M->N
x_auth_code = "test"

result_text = ""
uploaded_image_path = "static/uploaded_img.jpeg"


@app.route('/')
def index():
    return render_template('main.html')


@app.route('/predict-pneumonia-xray', methods=['POST', 'GET'])
def predict_pneumonia():
    file = request.files['file']
    global result_text
    # Assuming the file is an image, you can use PIL to open it
    image = Image.open(file)
    image.save("static/uploaded_img.jpeg")
    global uploaded_image_path
    uploaded_image_path = "static/uploaded_img.jpeg"

    result = predict_image_classification_sample(
    project="293924083198",
    endpoint_id="7163214900867629056",
    location="us-central1",
    filename="YOUR_IMAGE_FILE"
    )

    if "error_msg" not in result:
        json_data = json.loads(result)
        predicted_label = json_data['predicted_label']
        result_text = "AI Model Prediction: "
        # Find the entry for the predicted label in the 'scores' list
        for label, score in json_data['scores']:
            if label == predicted_label:
                score = float(score)
                score = round(score, 2)
                result_text += "%" + str(score * 100) + " percent chance -> " + label + ". "
                # Convert the score to a float
                score = float(score)
                additional_text = ""
                if predicted_label == "pneumonia":
                    if (score * 100) < 75:
                        additional_text = "You should see a doctor."
                    else:
                        additional_text = "Immediately seek medical help!"

                    result_text += additional_text

                    break

    # Pass the dynamic parameters directly to the 'results' route
    return redirect(url_for('results'))


@app.route('/predict-sars-cov-2-tomography', methods=['POST', 'GET'])
def predict_sars_cov_2():
    file = request.files['file']
    global result_text
    # Assuming the file is an image, you can use PIL to open it
    image = Image.open(file)
    im_saved = image.save("static/uploaded_img.jpeg")
    global uploaded_image_path
    uploaded_image_path = "static/uploaded_img.jpeg"

    result = predict_image_classification_sample(
        project="293924083198",
        endpoint_id="1924543371817254912",
        location="us-central1",
        filename="YOUR_IMAGE_FILE"
    )

    if "error_msg" not in result:
        json_data = json.loads(result)
        predicted_label = json_data['predicted_label']
        result_text = "AI Model Prediction: "
        # Find the entry for the predicted label in the 'scores' list
        for label, score in json_data['scores']:
            if label == predicted_label:
                score = float(score)
                score = round(score, 2)
                if label == "COVID":
                    result_text += "%" + str(score * 100) + " percent chance -> " + label.capitalize() + "."
                else:
                    result_text += "%" + str(score * 100) + " percent chance NOT COVID."

                # Convert the score to a float
                score = float(score)
                additional_text = ""
                if predicted_label == "covid":
                    if (score * 100) < 75:
                        additional_text = "You should see a doctor."
                    else:
                        additional_text = "Immediately seek medical help!"

                    result_text += additional_text

                    break

    # Pass the dynamic parameters directly to the 'results' route
    return redirect(url_for('results'))


@app.route('/predict-brain-tumor-mri', methods=['POST', 'GET'])
def predict_brain_tumor_mri():
    file = request.files['file']
    global x_auth_code
    global result_text
    # Assuming the file is an image, you can use PIL to open it
    image = Image.open(file)
    image.save("static/uploaded_img.jpeg")
    global uploaded_image_path
    uploaded_image_path = "static/uploaded_img.jpeg"

    result = predict_image_classification_sample(
        project="293924083198",
        endpoint_id="5549800334362148864",
        location="us-central1",
        filename=uploaded_image_path
    )

    if "error_msg" not in result:
        json_data = json.loads(result)
        predicted_label = json_data['predicted_label']
        result_text = "AI Model Prediction: "
        # Find the entry for the predicted label in the 'scores' list
        for label, score in json_data['scores']:
            if label == predicted_label:
                score = float(score)
                score = round(score, 2)

                if label == "glioma_tumor":

                    result_text += "%" + str(score * 100) + " chance -> Glioma tumor has found."
                elif label == "no_tumor":

                    result_text += "%" + str(score * 100) + " chance -> No tumor found."
                elif label == "meningioma_tumor":

                    result_text += "%" + str(score * 100) + " chance -> Meningioma tumor has found."
                elif label == "pituitary_tumor":

                    result_text += "%" + str(score * 100) + " chance -> Pituitary tumor has found."

                break

    # Pass the dynamic parameters directly to the 'results' route
    return redirect(url_for('results'))


@app.route('/results')
def results(dynamic_text=result_text, img_path=uploaded_image_path):
    return render_template('result.html', dynamic_text=result_text, img_path=uploaded_image_path)


# [START aiplatform_predict_image_classification_sample]
import base64

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict


def predict_image_classification_sample(
        project: str,
        endpoint_id: str,
        filename: str,
        location: str = "us-central1",
        api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    with open(filename, "rb") as f:
        file_content = f.read()

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5,
        max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/image_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))
    return predictions


# [END aiplatform_predict_image_classification_sample]


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
