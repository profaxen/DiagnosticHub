import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import time
import datetime
import plotly.graph_objects as go
import plotly.express as px
from src.predict import load_and_preprocess_image
import gdown

# --- GOOGLE DRIVE DOWNLOAD LOGIC ---
@st.cache_resource
def download_models_from_drive():
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    advanced_path = os.path.join(model_dir, 'advanced_best.h5')
    baseline_path = os.path.join(model_dir, 'baseline_best.h5')
    
    # Advanced Model ID: 1u4WaQNN-tHTAqJsOkkjAyhwrstBrfm83
    # Baseline Model ID: 1AXS9Z-RRVEl2tP4ZORw9RnKUM0hV2zKK
    
    if not os.path.exists(advanced_path):
        with st.spinner("Downloading Advanced Engine from Cloud (250MB)..."):
            url = f'https://drive.google.com/uc?id=1u4WaQNN-tHTAqJsOkkjAyhwrstBrfm83'
            gdown.download(url, advanced_path, quiet=False)
            
    if not os.path.exists(baseline_path):
        with st.spinner("Downloading Baseline Engine from Cloud (120MB)..."):
            url = f'https://drive.google.com/uc?id=1AXS9Z-RRVEl2tP4ZORw9RnKUM0hV2zKK'
            gdown.download(url, baseline_path, quiet=False)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PneumoScan Pro | Clinical Intelligence",
    page_icon="🔬",
    layout="wide",
)

# Trigger download before anything else
download_models_from_drive()

# --- LUXURY DESIGN SYSTEM v4.0 (Multi-Module Product) ---
st.markdown("""
<style>
    /* Clean UI Overrides */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    
    .stApp {
        background: #Fcfcfd;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Professional Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    [data-testid="stSidebar"] * {
        color: #F1F5F9 !important;
    }

    /* Navigation Radio Styling */
    .st-emotion-cache-1647967 {
        background-color: #1E293B !important;
        border-radius: 10px;
        padding: 5px;
    }

    /* Luxury Container */
    .product-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 24px;
        padding: 32px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
        margin-bottom: 24px;
    }

    /* Branding */
    .h1-brand {
        font-size: 3rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -1.5px;
        margin-bottom: 0;
    }
    
    .subtitle-brand {
        color: #64748B;
        font-size: 0.9rem;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 30px;
    }

    /* Metrics and Banners */
    .banner {
        padding: 20px;
        border-radius: 16px;
        font-weight: 700;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    .banner-pos { background: #FEF2F2; color: #991B1B; border: 1px solid #FEE2E2; }
    .banner-neg { background: #F0FDF4; color: #166534; border: 1px solid #DCFCE7; }

    /* Fixes */
    .block-container { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# --- UTILITIES ---
@st.cache_resource
def load_main_engine():
    model = tf.keras.models.load_model('models/advanced_best.h5')
    model(np.zeros((1, 224, 224, 3)))
    return model

def get_mapping(model, img_array):
    resnet = model.get_layer('resnet50v2')
    last_conv = next(l.name for l in reversed(resnet.layers) if isinstance(l, tf.keras.layers.Conv2D))
    grad_model = tf.keras.models.Model([resnet.input], [resnet.get_layer(last_conv).output, resnet.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)[0]
    output = conv_out[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(output, weights)
    cam = cv2.resize(cam, (500, 500))
    cam = np.maximum(cam, 0)
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)

def generate_report(prediction, confidence, risk):
    status = "POSITIVE" if prediction > 0.5 else "NEGATIVE"
    diagnosis = "Pneumonia Detected" if prediction > 0.5 else "Normal Lung Findings"
    
    report_text = f"""
===========================================================
             PNEUMOSCAN PRO: CLINICAL REPORT
===========================================================
GENERATED ON: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
REPORT ID:    PS-{datetime.datetime.now().strftime("%y%m%d%H%M")}
-----------------------------------------------------------

[SUMMARY OF FINDINGS]
---------------------
PATIENT STATUS:    {status}
PRIMARY DIAGNOSIS: {diagnosis}
AI CONFIDENCE:     {confidence:.2%}
RISK CATEGORY:     {risk}

[CLINICAL INTERPRETATION]
-------------------------
The AI analysis detected radiographic markers consistent with
{'pulmonary consolidation and opacities typical of Pneumonia.' if prediction > 0.5 else 'clear lung fields with no significant pathological markers.'}
The Neural Attention Mapping (Grad-CAM) should be reviewed to
confirm the anatomical location of these findings.

[TECHNICAL SPECIFICATIONS]
--------------------------
Neural Backbone:   ResNet50V2 (Transfer Learning)
Engine Version:    v2.5.0 (Proprietary)
Architect:         ADARSH

-----------------------------------------------------------
DISCLAIMER: This is an AI-generated decision support report. 
Final clinical diagnosis must be verified by a certified 
radiologist. This report is for research and educational 
purposes only.
===========================================================
    """
    return report_text.strip()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063206.png", width=60)
    st.markdown("### PNEUMOSCAN CONTROL")
    
    page = st.selectbox(
        "NAVIGATION",
        ["DIAGNOSTIC DASHBOARD", "SYSTEM ANALYTICS", "DATASET INTELLIGENCE", "ARCHITECT SPECS"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### CLINICAL STATUS")
    st.success("AI Core: **Operational**")
    st.info("Version: **2.5.0**")
    
    st.markdown("---")
    st.caption("PROPRIETARY SYSTEM")
    st.caption("DESIGNED BY ADARSH")

# --- PAGE: DIAGNOSTIC DASHBOARD ---
if page == "DIAGNOSTIC DASHBOARD":
    st.markdown('<h1 class="h1-brand">Diagnostic Hub.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-brand">Real-time Clinical Radiography Analysis</p>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.2], gap="large")

    with c1:
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        st.markdown("#### 01. Radiograph Input")
        file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if file:
            img = Image.open(file).convert("RGB")
            display_img = img.resize((500, 500))
            st.image(display_img, use_container_width=True, caption="Active Patient Radiograph")
            img.save("temp_buffer.jpg")
            
            if st.button("RUN CLINICAL ANALYSIS"):
                st.session_state['analyze_v4'] = True
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        if file and st.session_state.get('analyze_v4'):
            with st.spinner("Processing..."):
                model = load_main_engine()
                processed_img = load_and_preprocess_image("temp_buffer.jpg")
                score = model.predict(processed_img, verbose=0)[0][0]
                risk = "CRITICAL" if score > 0.7 else "MODERATE" if score > 0.4 else "LOW"
            
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            st.markdown("#### 02. Neural Verdict")
            
            if score > 0.5:
                st.markdown(f'<div class="banner banner-pos">PNEUMONIA DETECTED ({(score*100):.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="banner banner-neg">LUNGS CLEAR ({(100 - score*100):.1f}%)</div>', unsafe_allow_html=True)
            
            m1, m2 = st.columns(2)
            m1.metric("Engine Confidence", f"{score if score > 0.5 else 1-score:.2%}")
            m2.metric("Pathological Risk", risk)
            
            st.markdown("---")
            st.markdown("#### Clinical Report")
            conf = score if score > 0.5 else 1-score
            report_text = generate_report(score, conf, risk)
            st.download_button("DOWNLOAD OFFICIAL REPORT", report_text, file_name=f"Report_{datetime.datetime.now().strftime('%y%m%d%H%M')}.txt")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("<div style='height:450px; display:flex; align-items:center; justify-content:center; color:#94A3B8; border:2px dashed #E2E8F0; border-radius:24px;'>Awaiting Clinical Input...</div>", unsafe_allow_html=True)

    if file and st.session_state.get('analyze_v4'):
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        st.markdown("#### 03. Neural Attention Mapping")
        st.write("Spatial visualization of radiographic markers detected by the AI.")
        v1, v2 = st.columns(2)
        with v1:
            st.image(display_img, use_container_width=True, caption="Radiograph")
        with v2:
            heatmap = get_mapping(model, processed_img)
            orig = cv2.imread("temp_buffer.jpg")
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            orig = cv2.resize(orig, (500, 500))
            hm = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (500, 500))), cv2.COLORMAP_JET)
            hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig, 0.6, hm, 0.4, 0)
            st.image(overlay, use_container_width=True, caption="AI Pathological Mapping")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: SYSTEM ANALYTICS ---
elif page == "SYSTEM ANALYTICS":
    st.markdown('<h1 class="h1-brand">Performance Metrics.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-brand">Comparative Analysis of Diagnostic Engines</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        st.markdown("#### Diagnostic Accuracy (Advanced)")
        if os.path.exists('outputs/plots/advanced_history.png'):
            st.image('outputs/plots/advanced_history.png', use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        st.markdown("#### Confusion Matrix (Pro Engine)")
        if os.path.exists('outputs/confusion_matrices/advanced_cm.png'):
            st.image('outputs/confusion_matrices/advanced_cm.png', use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="product-card">', unsafe_allow_html=True)
    st.markdown("#### Statistical Comparison")
    df_metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Baseline CNN': [0.90, 0.91, 0.93, 0.92],
        'Advanced ResNet': [0.93, 0.92, 0.98, 0.95]
    }
    st.table(df_metrics)
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: DATASET INTELLIGENCE ---
elif page == "DATASET INTELLIGENCE":
    st.markdown('<h1 class="h1-brand">Data Insights.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-brand">Analysis of 5,856 Chest Radiographs</p>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        st.markdown("#### Class Distribution")
        # Custom Plotly Pie
        fig = px.pie(values=[1341, 3875], names=['Normal', 'Pneumonia'], color_discrete_sequence=['#E2E8F0', '#0F172A'])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        st.markdown("#### Training Challenges")
        st.write("- **Class Imbalance:** 3:1 Pneumonia/Normal ratio.")
        st.write("- **Noise:** Presence of medical artifacts (wires, tubes).")
        st.write("- **Resolution:** High variability in image dimensions.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: ARCHITECT SPECS ---
else:
    st.markdown('<h1 class="h1-brand">Architecture.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-brand">The Engineering Behind PneumoScan Pro</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="product-card">', unsafe_allow_html=True)
    st.markdown("#### Tech Stack")
    st.code("""
    Language: Python 3.11
    Deep Learning: TensorFlow 2.10+
    Backbone: ResNet50V2 (Transfer Learning)
    Optimization: Adam (LR: 1e-4)
    Explainability: Grad-CAM (Gradient-weighted Class Activation Mapping)
    """)
    st.markdown("#### Key Innovations")
    st.write("1. **Weighted Cross-Entropy:** Solved the 3:1 data imbalance.")
    st.write("2. **Feature Fusion:** Leveraging ImageNet weights for low-level texture detection.")
    st.write("3. **Real-time Explainability:** Integrated heatmapping for clinical trust.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown(f"""
    <div style="text-align: center; padding: 60px; color: #94A3B8; font-size: 0.8rem; letter-spacing: 2px;">
        PNEUMOSCAN PRO • DESIGNED & ARCHITECTED BY <span style="color:#0F172A; font-weight:800;">ADARSH</span> • © 2026
    </div>
""", unsafe_allow_html=True)
