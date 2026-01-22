# Medical Insurance Cost Prediction

A machine learning project that predicts annual medical insurance charges based on personal information using Random Forest Regressor.

## Project Overview

This project implements a complete ML pipeline to predict medical insurance costs based on factors like age, BMI, smoking status, number of children, sex, and region.

### Key Features
- **Comprehensive preprocessing pipeline** with 5 distinct steps
- **Random Forest model** optimized through GridSearchCV
- **Interactive Gradio interface** for easy predictions
- **High accuracy** with R² score of ~0.84

## Technologies Used

- Python 3.10+
- scikit-learn
- pandas
- numpy
- Gradio
- Google Colab

## Project Structure

```
medical-insurance-prediction/
│
├── app.py                          # Gradio deployment app
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── main.ipynb                      # Complete ML pipeline notebook
├── best_model.pkl                  # Trained model (generated)
└── insurance.csv                   # Dataset
```

## Quick Start

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/ishtiaksaad/medical-insurance-prediction.git
cd medical-insurance-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Gradio app:
```bash
python app.py
```

## Dataset

The dataset contains 1,338 records with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Age of the insured person |
| sex | Categorical | Gender (male/female) |
| bmi | Numeric | Body Mass Index |
| children | Numeric | Number of dependents |
| smoker | Categorical | Smoking status (yes/no) |
| region | Categorical | Geographic region (northeast, northwest, southeast, southwest) |
| charges | Numeric | **Target**: Annual medical insurance costs |

## ML Pipeline

### 1. Data Preprocessing (5 Steps)
1. **Missing Value Check** - Verified no missing values
2. **Feature Separation** - Split features and target variable
3. **Column Type Identification** - Identified numerical and categorical columns
4. **Outlier Handling** - Applied IQR capping on target variable
5. **Feature Engineering** - Created binary smoker feature

### 2. Model Development
- **Algorithm**: Random Forest Regressor
- **Justification**: Handles non-linear relationships, robust to outliers, provides feature importance
- **Hyperparameters Tuned**:
  - `n_estimators`: [100, 200]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5]

### 3. Best Model Configuration
```python
RandomForestRegressor(
    max_depth=10,
    min_samples_split=5,
    n_estimators=200,
    random_state=42
)
```

## Usage Example

```python
# Input features
age = 35
sex = "male"
bmi = 28.5
children = 2
smoker = "no"
region = "northwest"

# Prediction
predicted_cost = model.predict(input_data)
# Output: $5,234.67
```

## Live Demo

Try the live demo on Hugging Face Spaces:
[Medical Insurance Predictor](https://huggingface.co/spaces/ishtiaksaad/medical-insurance-predictor)

## Model Insights

The model identified the following key factors affecting insurance costs:
1. **Smoking status** - Most significant predictor
2. **Age** - Positive correlation with costs
3. **BMI** - Higher BMI increases costs
4. **Number of children** - Moderate impact
5. **Region** - Minor variations across regions

## Future Improvements

- [ ] Implement additional models (XGBoost, LightGBM)
- [ ] Add SHAP values for interpretability
- [ ] Create feature importance visualization
- [ ] Expand dataset with more features
- [ ] Add batch prediction capability

## License

This project is open source and available under the MIT License.