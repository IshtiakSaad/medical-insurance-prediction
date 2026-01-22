import gradio as gr
import pandas as pd
import pickle
import os
import warnings
import logging
import sklearn

# Minimal logging
logging.basicConfig(filename="prediction.log", level=logging.INFO)

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.InconsistentVersionWarning)

MODEL_PATH = "best_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Metadata check
if hasattr(model, "metadata"):
    expected_features = {"age", "sex", "bmi", "children", "smoker", "region"}
    if set(model.metadata.get("features", [])) != expected_features:
        logging.warning("Model features differ from expected features.")

def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    if None in [age, sex, bmi, children, smoker, region]:
        return "Please fill all fields."

    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    try:
        pred = model.predict(input_df)[0]
        logging.info(f"Predicted {pred} for inputs {input_df.to_dict(orient='records')[0]}")
        return f"${pred:,.2f}"
    except Exception:
        logging.exception("Prediction failed")
        return "Prediction error."

# Gradio UI
interface = gr.Interface(
    fn=predict_insurance_cost,
    inputs=[
        gr.Number(label="Age", minimum=18, maximum=100),
        gr.Dropdown(["male", "female"], label="Sex"),
        gr.Number(label="BMI", minimum=15, maximum=55),
        gr.Number(label="Number of Children", minimum=0, maximum=5),
        gr.Dropdown(["yes", "no"], label="Smoker"),
        gr.Dropdown(["southeast", "southwest", "northwest", "northeast"], label="Region")
    ],
    outputs=gr.Textbox(label="Estimated Annual Cost"),
    examples=[
        [19, "female", 27.9, 0, "yes", "southwest"],
        [35, "male", 28.5, 2, "no", "northwest"]
    ],
    title="Insurance Cost Predictor"
)

if __name__ == "__main__":
    # interface.launch(server_name="0.0.0.0", server_port=7860)
    interface.launch()
