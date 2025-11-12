import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #2E86AB;
        padding-bottom: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .negative {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    models = {
        'diabetes': {
            'model': joblib.load('models/diabetes_model.pkl'),
            'scaler': joblib.load('models/diabetes_scaler.pkl'),
            'results': joblib.load('models/diabetes_results.pkl')
        },
        'heart': {
            'model': joblib.load('models/heart_model.pkl'),
            'scaler': joblib.load('models/heart_scaler.pkl'),
            'results': joblib.load('models/heart_results.pkl')
        },
        'cancer': {
            'model': joblib.load('models/cancer_model.pkl'),
            'scaler': joblib.load('models/cancer_scaler.pkl'),
            'results': joblib.load('models/cancer_results.pkl'),
            'features': joblib.load('models/cancer_feature_names.pkl')
        }
    }
    return models

try:
    models = load_models()
    models_loaded = True
except:
    models_loaded = False
    st.error("Models not found! Please run 'python3 train_models.py' first.")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    
    page = st.selectbox(
        "Select Disease Prediction",
        ["Home", "Diabetes Prediction", "Heart Disease Prediction", "Breast Cancer Prediction", "Model Comparison"]
    )
    
    st.markdown("---")
    
    st.markdown("### About")
    st.info("""
    This application uses Machine Learning to predict various diseases based on medical parameters.
    
    **Models Used:**
    - Random Forest
    - Logistic Regression
    - XGBoost
    - Gradient Boosting
    """)
    
    st.markdown("---")
    st.markdown("### Developer")
    st.markdown("**Priyanka Malavade**")
    st.markdown("BCA Graduate 2024")
    st.markdown("Data Science Portfolio - Project 2")

# HOME PAGE
if page == "Home":
    st.title("Disease Prediction System")
    st.markdown("### AI-Powered Medical Diagnosis Assistant")
    
    st.markdown("""
    Welcome to the Disease Prediction System. This application uses advanced Machine Learning algorithms 
    to predict the likelihood of various diseases based on medical parameters.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Diabetes Prediction")
        st.markdown("""
        Predicts diabetes risk based on:
        - Glucose levels
        - BMI
        - Age
        - Blood pressure
        - And more...
        """)
        
    with col2:
        st.markdown("### Heart Disease Prediction")
        st.markdown("""
        Assesses heart disease risk using:
        - Chest pain type
        - Cholesterol levels
        - Blood pressure
        - ECG results
        - And more...
        """)
        
    with col3:
        st.markdown("### Breast Cancer Prediction")
        st.markdown("""
        Analyzes cancer risk based on:
        - Cell characteristics
        - Tumor measurements
        - Texture features
        - And more...
        """)
    
    st.markdown("---")
    
    if models_loaded:
        st.success("All models loaded successfully! Select a prediction type from the sidebar.")
        
        # Display model performance
        st.markdown("### Model Performance Summary")
        
        performance_data = {
            'Disease': ['Diabetes', 'Heart Disease', 'Breast Cancer'],
            'Best Model': [
                list(models['diabetes']['results'].keys())[0],
                list(models['heart']['results'].keys())[0],
                list(models['cancer']['results'].keys())[0]
            ],
            'Accuracy': [
                models['diabetes']['results'][list(models['diabetes']['results'].keys())[0]]['accuracy'],
                models['heart']['results'][list(models['heart']['results'].keys())[0]]['accuracy'],
                models['cancer']['results'][list(models['cancer']['results'].keys())[0]]['accuracy']
            ]
        }
        
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance, hide_index=True, use_container_width=True)

# DIABETES PREDICTION
elif page == "Diabetes Prediction":
    st.title("Diabetes Prediction")
    st.markdown("### Enter patient medical information")
    
    if not models_loaded:
        st.stop()
    
    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        
        with col2:
            insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
        
        submitted = st.form_submit_button("Predict Diabetes Risk", type="primary")
    
    if submitted:
        # Prepare input
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                               insulin, bmi, dpf, age]])
        
        # Scale input
        input_scaled = models['diabetes']['scaler'].transform(input_data)
        
        # Make prediction
        prediction = models['diabetes']['model'].predict(input_scaled)[0]
        probability = models['diabetes']['model'].predict_proba(input_scaled)[0]
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.markdown(f'<div class="prediction-box positive"><h3>High Risk of Diabetes</h3><p>Probability: {probability[1]:.2%}</p></div>', 
                       unsafe_allow_html=True)
            st.warning("The model predicts a high risk of diabetes. Please consult a healthcare professional.")
        else:
            st.markdown(f'<div class="prediction-box negative"><h3>Low Risk of Diabetes</h3><p>Probability: {probability[0]:.2%}</p></div>', 
                       unsafe_allow_html=True)
            st.success("The model predicts a low risk of diabetes. Maintain a healthy lifestyle!")
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Diabetes Risk Probability (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if prediction == 1 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# HEART DISEASE PREDICTION
elif page == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    st.markdown("### Enter patient cardiac information")
    
    if not models_loaded:
        st.stop()
    
    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
            chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
                             format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                                 format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x])
            thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col3:
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2],
                               format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
            ca = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3, 4])
            thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                              format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])
        
        submitted = st.form_submit_button("Predict Heart Disease Risk", type="primary")
    
    if submitted:
        # Prepare input
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                               thalach, exang, oldpeak, slope, ca, thal]])
        
        # Scale input
        input_scaled = models['heart']['scaler'].transform(input_data)
        
        # Make prediction
        prediction = models['heart']['model'].predict(input_scaled)[0]
        probability = models['heart']['model'].predict_proba(input_scaled)[0]
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.markdown(f'<div class="prediction-box positive"><h3>High Risk of Heart Disease</h3><p>Probability: {probability[1]:.2%}</p></div>', 
                       unsafe_allow_html=True)
            st.warning("The model predicts a high risk of heart disease. Please consult a cardiologist immediately.")
        else:
            st.markdown(f'<div class="prediction-box negative"><h3>Low Risk of Heart Disease</h3><p>Probability: {probability[0]:.2%}</p></div>', 
                       unsafe_allow_html=True)
            st.success("The model predicts a low risk of heart disease. Keep up the healthy habits!")
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk Probability (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if prediction == 1 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# BREAST CANCER PREDICTION
elif page == "Breast Cancer Prediction":
    st.title("Breast Cancer Prediction")
    st.markdown("### Enter tumor characteristics")
    
    if not models_loaded:
        st.stop()
    
    st.info("This is a simplified interface. In practice, these measurements come from medical imaging.")
    
    with st.form("cancer_form"):
        st.markdown("#### Mean Values")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=50.0, value=14.0, step=0.1)
            texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=50.0, value=19.0, step=0.1)
            perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=92.0, step=0.1)
            area_mean = st.number_input("Area Mean", min_value=0.0, max_value=3000.0, value=655.0, step=1.0)
            smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=1.0, value=0.096, step=0.001)
        
        with col2:
            compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=1.0, value=0.104, step=0.001)
            concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=1.0, value=0.089, step=0.001)
            concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=1.0, value=0.048, step=0.001)
            symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=1.0, value=0.181, step=0.001)
            fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, max_value=1.0, value=0.063, step=0.001)
        
        with col3:
            radius_se = st.number_input("Radius SE", min_value=0.0, max_value=5.0, value=0.406, step=0.001)
            texture_se = st.number_input("Texture SE", min_value=0.0, max_value=5.0, value=1.217, step=0.001)
            perimeter_se = st.number_input("Perimeter SE", min_value=0.0, max_value=50.0, value=2.866, step=0.001)
            area_se = st.number_input("Area SE", min_value=0.0, max_value=500.0, value=40.34, step=0.1)
            smoothness_se = st.number_input("Smoothness SE", min_value=0.0, max_value=1.0, value=0.007, step=0.0001)
        
        # For simplicity, we'll use default values for worst features
        st.markdown("#### Worst Values (using defaults)")
        
        submitted = st.form_submit_button("Predict Cancer Risk", type="primary")
    
    if submitted:
        # Create full feature array (30 features total)
        # Using provided values for mean and SE, defaults for worst
        input_data = np.array([[
            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
            compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
            radius_se, texture_se, perimeter_se, area_se, smoothness_se,
            0.025, 0.031, 0.029, 0.021, 0.004,  # SE values (defaults)
            16.0, 25.0, 107.0, 880.0, 0.132,  # Worst values (defaults)
            0.254, 0.272, 0.114, 0.290, 0.084
        ]])
        
        # Scale input
        input_scaled = models['cancer']['scaler'].transform(input_data)
        
        # Make prediction
        prediction = models['cancer']['model'].predict(input_scaled)[0]
        probability = models['cancer']['model'].predict_proba(input_scaled)[0]
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        if prediction == 0:
            st.markdown(f'<div class="prediction-box positive"><h3>Malignant (Cancerous)</h3><p>Probability: {probability[0]:.2%}</p></div>', 
                       unsafe_allow_html=True)
            st.warning("The model predicts malignant tumor. Please consult an oncologist immediately.")
        else:
            st.markdown(f'<div class="prediction-box negative"><h3>Benign (Non-cancerous)</h3><p>Probability: {probability[1]:.2%}</p></div>', 
                       unsafe_allow_html=True)
            st.success("The model predicts benign tumor. Regular monitoring is still recommended.")
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[0] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Malignant Probability (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if prediction == 0 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# MODEL COMPARISON
elif page == "Model Comparison":
    st.title("Model Performance Comparison")
    st.markdown("### Compare different ML algorithms")
    
    if not models_loaded:
        st.stop()
    
    disease = st.selectbox("Select Disease", ["Diabetes", "Heart Disease", "Breast Cancer"])
    
    if disease == "Diabetes":
        results = models['diabetes']['results']
    elif disease == "Heart Disease":
        results = models['heart']['results']
    else:
        results = models['cancer']['results']
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    st.dataframe(df_comparison, hide_index=True, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(df_comparison, x='Model', y='Accuracy', 
                     title='Model Accuracy Comparison',
                     color='Accuracy',
                     color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(df_comparison, x='Model', y='ROC-AUC',
                     title='Model ROC-AUC Comparison',
                     color='ROC-AUC',
                     color_continuous_scale='plasma')
        st.plotly_chart(fig, use_container_width=True)
    
    # Metrics comparison
    st.markdown("### All Metrics Comparison")
    
    metrics_df = df_comparison.set_index('Model').T
    fig = px.imshow(metrics_df, 
                    labels=dict(x="Model", y="Metric", color="Score"),
                    x=metrics_df.columns,
                    y=metrics_df.index,
                    color_continuous_scale='RdYlGn',
                    aspect="auto")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Disclaimer
st.markdown("---")
st.markdown("""
**Disclaimer:** This is a demonstration application for educational purposes. 
Always consult qualified healthcare professionals for medical diagnosis and treatment.
""")

if __name__ == "__main__":
    pass