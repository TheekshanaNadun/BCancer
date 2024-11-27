import gradio as gr
import joblib

# Load your model
model = joblib.load("model.pkl")

# Define a function that makes predictions using the model
def predict(input_data):
    # Perform any necessary preprocessing here
    result = model.predict([input_data])  # Adjust this based on your model
    return result

# Create Gradio interface
iface = gr.Interface(fn=predict, 
                     inputs="text",  # You can change the input type as per your needs (e.g., image, text, etc.)
                     outputs="text")  # Specify the type of output (text, image, etc.)

# Launch the app
iface.launch(share=True)  # share=True allows others to access your app with a public link
