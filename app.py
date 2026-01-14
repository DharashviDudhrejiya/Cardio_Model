import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
bundle = joblib.load("cardio_model.pkl")
model = bundle['model']
feature_order = bundle['columns']
scaler_hi = bundle['scaler_hi']
scaler_lo = bundle['scaler_lo']

st.markdown("""
    <style>
    /* Light beige background */
    .stApp {
        background: #f5f1e8;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Title styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #2c3e50;
        text-align: left;
        margin-bottom: 0.3rem;
        font-family: Georgia, serif;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: left;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-style: italic;
    }
    
    /* Input labels */
    .stSelectbox label, .stNumberInput label {
        color: #34495e !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Number input styling */
    .stNumberInput input {
        background: #ffffff !important;
        border: 2px solid #d4c5b9 !important;
        border-radius: 4px !important;
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stNumberInput input:focus {
        border-color: #8b7355 !important;
        box-shadow: 0 0 0 2px rgba(139, 115, 85, 0.1) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 2px solid #d4c5b9 !important;
        border-radius: 4px !important;
        color: #000000 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Form container */
    .stForm {
        background: #fdfcfa;
        border-radius: 8px;
        padding: 2.5rem;
        border: 3px solid #d4c5b9;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Submit Button */
    .stButton > button {
        background: #8b7355;
        color: #ffffff;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.9rem 2.5rem;
        border-radius: 4px;
        border: none;
        width: 100%;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: #6d5943;
        box-shadow: 0 4px 12px rgba(139, 115, 85, 0.3);
    }
    
    /* Result cards */
    .result-card {
        padding: 2.5rem;
        border-radius: 8px;
        text-align: left;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 2rem 0;
        border: 3px solid;
    }
    
    .high-risk {
        background: #fff5f5;
        border-color: #c0392b;
        color: #c0392b;
    }
    
    .low-risk {
        background: #f0f9f4;
        border-color: #27ae60;
        color: #27ae60;
    }
    
    /* Metric boxes */
    .metric-box {
        background: #ffffff;
        border-radius: 6px;
        padding: 1.8rem;
        margin: 0.5rem;
        text-align: left;
        border: 2px solid #e8dfd5;
    }
    
    .metric-label {
        color: #7f8c8d;
        font-size: 0.75rem;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .metric-value {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 800;
    }
    
    /* Section headers */
    .section-header {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 2.5rem 0 1.2rem 0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        border-bottom: 3px solid #8b7355;
        padding-bottom: 0.5rem;
    }
    
    /* Info box */
    .info-box {
        background: #fff9f0;
        border: 2px solid #f39c12;
        padding: 1.2rem;
        border-radius: 6px;
        margin: 1.5rem 0;
        color: #7f8c8d;
        font-size: 0.95rem;
    }
    
    /* Radio buttons */
    .stRadio > div {
        flex-direction: row;
        gap: 1rem;
        justify-content: flex-start;
    }
    
    .stRadio > div > label {
        background: #ffffff;
        border: 2px solid #d4c5b9;
        border-radius: 4px;
        padding: 0.8rem 1.8rem;
        color: #000000;
        font-weight: 700;
        font-size: 0.95rem;
    }
    
    .stRadio > div > label:hover {
        border-color: #8b7355;
        background: #fdfcfa;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: #8b7355;
        border-color: #8b7355;
        color: #ffffff;
    }
    
    .stRadio label > div {
        color: #000000 !important;
    }
    
    .stRadio label[data-checked="true"] > div {
        color: #ffffff !important;
    }
    
    /* Checkbox styling */
    .stCheckbox {
        padding: 1rem;
        background: #ffffff;
        border-radius: 6px;
        border: 2px solid #d4c5b9;
    }
    
    .stCheckbox > label {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    
    .stCheckbox input[type="checkbox"] {
        width: 22px;
        height: 22px;
        accent-color: #8b7355;
        cursor: pointer;
    }
    
    .stCheckbox > label > div {
        color: #000000 !important;
    }
    
    .stCheckbox span {
        color: #000000 !important;
    }
    
    /* Subsection headers */
    h4 {
        color: #34495e !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        margin: 1.8rem 0 0.8rem 0 !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    /* Tab label */
    .tab-label {
        color: #34495e;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
        display: block;
        text-align: left;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="Cardio Risk Predictor", page_icon="❤️", layout="centered")

# Header
st.markdown('<h1 class="main-title">Heart Health Assessment</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine learning-based cardiovascular risk evaluation</p>', unsafe_allow_html=True)

# Info box
st.markdown("""
    <div class="info-box">
        <strong>About this assessment:</strong> Provide your health information below for an AI-driven cardiovascular risk analysis 
        based on extensive clinical data patterns.
    </div>
""", unsafe_allow_html=True)

# --- Input form ---
st.markdown('<div class="section-header">Health Data Entry</div>', unsafe_allow_html=True)

with st.form("input_form"):
    # Personal Information
    st.markdown("#### Basic Information")
    col1, col2 = st.columns(2)
    age_years = col1.number_input("Age (years)", 20, 90, 50)
    gender = col2.selectbox("Gender", ["Male", "Female"])
    
    st.markdown("#### Body Measurements")
    col3, col4 = st.columns(2)
    height = col3.number_input("Height (cm)", 120, 220, 170)
    weight = col4.number_input("Weight (kg)", 40, 150, 70)
    
    st.markdown("#### Blood Pressure")
    col5, col6 = st.columns(2)
    ap_hi = col5.number_input("Systolic BP (mmHg)", 90, 250, 120)
    ap_lo = col6.number_input("Diastolic BP (mmHg)", 40, 180, 80)
    
    st.markdown("#### Laboratory Results")
    
    # Cholesterol with radio buttons
    st.markdown('<span class="tab-label">Cholesterol Level</span>', unsafe_allow_html=True)
    cholesterol = st.radio(
        "cholesterol_radio",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "High"}[x],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Glucose with radio buttons
    st.markdown('<span class="tab-label">Glucose Level</span>', unsafe_allow_html=True)
    gluc = st.radio(
        "glucose_radio",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "High"}[x],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("#### Lifestyle Information")
    
    # Toggle switches for lifestyle factors
    col7, col8, col9 = st.columns(3)
    
    with col7:
        smoke = st.checkbox("Smoking", value=False, key="smoke_toggle")
        smoke = 1 if smoke else 0
    
    with col8:
        alco = st.checkbox("Alcohol", value=False, key="alco_toggle")
        alco = 1 if alco else 0
    
    with col9:
        active = st.checkbox("Physical Activity", value=False, key="active_toggle")
        active = 1 if active else 0
    
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Calculate Risk")

# --- Prediction logic ---
if submitted:
    # Load scalers (now available in bundle)
    scaler_hi = bundle['scaler_hi']
    scaler_lo = bundle['scaler_lo']

    # Fix gender: 1=Female, 2=Male
    gender_val = 2 if gender == "Male" else 1

    # Base input DataFrame
    features_df = pd.DataFrame([{
        'gender': gender_val, 'height': height, 'weight': weight,
        'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol,
        'gluc': gluc, 'smoke': smoke, 'alco': alco, 'active': active,
        'age_years': age_years
    }])

    # Clip age to min=33 for OOD (young = low risk in data)
    if age_years < 33:
        features_df['age_years'] = 33

    # BMI
    features_df['bmi'] = features_df['weight'] / ((features_df['height'] / 100) ** 2)

    # EXACT training binning: pd.cut([0,40,50,60,np.inf], labels=[1,2,3,4])
    features_df['age_group'] = pd.cut(
        features_df['age_years'],
        bins=[0, 40, 50, 60, np.inf],
        labels=[1, 2, 3, 4],
        right=False
    ).astype(int)

    # EXACT z-scores using loaded scalers
    features_df['ap_hi_z'] = scaler_hi.transform(features_df[['ap_hi']]).flatten()
    features_df['ap_lo_z'] = scaler_lo.transform(features_df[['ap_lo']]).flatten()

    # Exact logs (np.log, as in notebook)
    features_df['ap_hi_log'] = np.log(features_df['ap_hi'])
    features_df['ap_lo_log'] = np.log(features_df['ap_lo'])
    features_df['bmi_log'] = np.log(features_df['bmi'])

    # Reindex to training columns
    features_df = features_df.reindex(columns=feature_order, fill_value=0)

    # Cast categoricals to int
    cat_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_group']
    for col in cat_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(int)

    # Predict with RF model
    prediction = model.predict(features_df)[0]
    proba = model.predict_proba(features_df)[0][1] * 100  # P(high risk)

    st.markdown('<div class="section-header">Risk Assessment Results</div>', unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown(f"""
            <div class="result-card high-risk">
                <strong>ELEVATED RISK DETECTED</strong><br>
                <span style="font-size: 2.5rem; font-weight: 800;">{proba:.1f}%</span><br>
                <span style="font-size: 0.85rem;">Cardiovascular Disease Probability</span>
            </div>
        """, unsafe_allow_html=True)
        st.warning("**Medical Consultation Advised:** Schedule an appointment with your healthcare provider for detailed evaluation.")
    else:
        st.markdown(f"""
            <div class="result-card low-risk">
                <strong>FAVORABLE RISK PROFILE</strong><br>
                <span style="font-size: 2.5rem; font-weight: 800;">{proba:.1f}%</span><br>
                <span style="font-size: 0.85rem;">Cardiovascular Disease Probability</span>
            </div>
        """, unsafe_allow_html=True)
        st.success("**Positive Indicators:** Your metrics suggest good cardiovascular health.")
    
    # Display calculated metrics
    bmi = features_df['bmi'].values[0]
    st.markdown('<div class="section-header">Calculated Health Metrics</div>', unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Body Mass Index</div>
                <div class="metric-value">{bmi:.1f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">BP Reading</div>
                <div class="metric-value">{ap_hi}/{ap_lo}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        bmi_category = "Normal" if 18.5 <= bmi < 25 else ("Underweight" if bmi < 18.5 else "Overweight")
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">BMI Category</div>
                <div class="metric-value" style="font-size: 1.2rem;">{bmi_category}</div>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #95a5a6; padding: 20px;">
        <small>Medical Disclaimer: This assessment tool provides informational guidance only and does not substitute professional medical consultation.</small>
    </div>
""", unsafe_allow_html=True)


# # streamlit run app.py