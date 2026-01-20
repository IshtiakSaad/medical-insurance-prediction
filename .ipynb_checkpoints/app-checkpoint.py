import gradio as gr
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_model():
    """
    This function recreates the model pipeline.
    In practice, you should load a saved model using pickle.
    """
    numerical_cols = ['age', 'bmi', 'children']
    categorical_cols = ['sex', 'smoker', 'region']
    
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            max_depth=10, 
            min_samples_split=5, 
            n_estimators=200, 
            random_state=42
        ))
    ])
    
    return model_pipeline

# Load or create model
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    model = create_model()
    # You'll need to fit this with your training data
    print("Warning: Model not loaded. Please upload trained model.")

def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    """
    Predict medical insurance costs based on input features.
    
    Parameters:
    - age: Age of the person
    - sex: Gender (male/female)
    - bmi: Body Mass Index
    - children: Number of dependents
    - smoker: Smoking status (yes/no)
    - region: Geographic region
    
    Returns:
    - Predicted annual insurance cost in USD
    """
    try:
        # Create input dataframe
        input_data = pd.DataFrame([{
            "age": int(age),
            "sex": sex,
            "bmi": float(bmi),
            "children": int(children),
            "smoker": smoker,
            "region": region
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Return formatted result
        return f"${prediction:,.2f}"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_insurance_cost,
    inputs=[
        gr.Number(label="Age", minimum=18, maximum=100, value=30),
        gr.Dropdown(["male", "female"], label="Sex", value="male"),
        gr.Number(label="BMI (Body Mass Index)", minimum=15, maximum=55, value=25),
        gr.Number(label="Number of Children", minimum=0, maximum=5, value=0),
        gr.Dropdown(["yes", "no"], label="Smoker", value="no"),
        gr.Dropdown(
            ["southeast", "southwest", "northwest", "northeast"], 
            label="Region", 
            value="southeast"
        )
    ],
    outputs=gr.Textbox(label="Predicted Annual Insurance Cost"),
    title="Medical Insurance Cost Prediction",
    description="""
    This model predicts annual medical insurance charges based on personal information.
    
    **How to use:**
    1. Enter your age (18-100)
    2. Select your sex
    3. Enter your BMI (typically 15-55)
    4. Enter number of children/dependents
    5. Indicate smoking status
    6. Select your region
    7. Click Submit to see the predicted cost
    
    **Model Details:**
    - Algorithm: Random Forest Regressor
    - R² Score: ~0.84
    - MAE: ~$2,191
    """,
    examples=[
        [19, "female", 27.9, 0, "yes", "southwest"],
        [35, "male", 28.5, 2, "no", "northwest"],
        [45, "female", 32.1, 3, "yes", "southeast"],
        [25, "male", 24.0, 0, "no", "northeast"],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()