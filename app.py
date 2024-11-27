import gradio as gr
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Define the prediction function
def predict(
    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
    fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
    smoothness_se, compactness_se, concavity_se
):
    # Prepare the input data as a numpy array (reshape to 2D array)
    input_data = np.array([
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
        fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
        smoothness_se, compactness_se, concavity_se
    ]).reshape(1, -1)  # Reshape for a single sample input
    
    # Make prediction using the model
    prediction = model.predict(input_data)
    
    # Decode the prediction result
    if prediction == 1:
        return "Malignant (Cancerous)", gr.Textbox.update(visible=True, value="Malignant (Cancerous)", interactive=False, color="red")
    else:
        return "Benign (Non-cancerous)", gr.Textbox.update(visible=True, value="Benign (Non-cancerous)", interactive=False, color="green")

# Define the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(0, 50, label="Radius Mean", step=0.1),
        gr.Slider(0, 50, label="Texture Mean", step=0.1),
        gr.Slider(0, 200, label="Perimeter Mean", step=0.1),
        gr.Slider(0, 1000, label="Area Mean", step=0.1),
        gr.Slider(0, 1, label="Smoothness Mean", step=0.001),
        gr.Slider(0, 1, label="Compactness Mean", step=0.001),
        gr.Slider(0, 1, label="Concavity Mean", step=0.001),
        gr.Slider(0, 1, label="Concave Points Mean", step=0.001),
        gr.Slider(0, 1, label="Symmetry Mean", step=0.001),
        gr.Slider(0, 1, label="Fractal Dimension Mean", step=0.001),
        gr.Slider(0, 50, label="Radius SE", step=0.1),
        gr.Slider(0, 50, label="Texture SE", step=0.1),
        gr.Slider(0, 200, label="Perimeter SE", step=0.1),
        gr.Slider(0, 1000, label="Area SE", step=0.1),
        gr.Slider(0, 1, label="Smoothness SE", step=0.001),
        gr.Slider(0, 1, label="Compactness SE", step=0.001),
        gr.Slider(0, 1, label="Concavity SE", step=0.001),
    ],
    outputs=[
        gr.Textbox(label="Prediction", interactive=False),
    ],
    title="Breast Cancer Prediction",
    description="Enter the features of a sample to predict whether it is malignant or benign.",
)

# Launch the app
iface.launch(share=True)
