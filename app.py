# ============================================================
#  Crop Irrigation Need Predictor — Streamlit UI
#  SDG 2 - Zero Hunger | SDG 15 - Life on Land
#  Features: Crop selector, Irrigation amount, Result history
# ============================================================

import streamlit as st
import pandas as pd
import joblib

# ============================================================
#  PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Irrigation Predictor",
    page_icon="🌱",
    layout="wide"
)

# ============================================================
#  LOAD MODEL & SCALER
# ============================================================

@st.cache_resource
def load_model():
    model  = joblib.load("model/irrigation_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ============================================================
#  SESSION STATE — for result history
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
#  CROP-SPECIFIC ADVICE DICTIONARY
# ============================================================

CROP_ADVICE = {
    "Rice"        : "Rice thrives in waterlogged conditions. Maintain consistent soil moisture and irrigate frequently during the growing season.",
    "Maize"       : "Maize is moderately water-sensitive. Irrigate during tasseling and silking stages — water stress at these points reduces yield significantly.",
    "Chickpea"    : "Chickpea is drought-tolerant. Over-irrigation can cause root rot — irrigate only when soil moisture is critically low.",
    "Kidney Beans": "Kidney beans need moderate, consistent moisture. Avoid waterlogging and irrigate during flowering and pod formation stages.",
    "Pigeon Peas" : "Pigeon peas are drought-resistant. Irrigate sparingly — excess water can damage roots and reduce nitrogen fixation.",
    "Moth Beans"  : "Moth beans are highly drought-tolerant. Minimal irrigation needed — only during prolonged dry spells.",
    "Mung Bean"   : "Mung bean requires moderate moisture. Irrigate lightly and avoid waterlogging, especially during the pod-filling stage.",
    "Black Gram"  : "Black gram needs moderate water. Irrigate at flowering and grain-filling stages for best yield outcomes.",
    "Lentil"      : "Lentils are fairly drought-tolerant. Irrigate during flowering and pod development, but avoid excess moisture.",
    "Pomegranate" : "Pomegranate is drought-tolerant once established. Drip irrigation is ideal — water stress during fruiting affects fruit quality.",
    "Banana"      : "Banana is a high water-demand crop. Irrigate frequently and maintain consistently moist soil throughout the growing cycle.",
    "Mango"       : "Mango trees need water stress before flowering to stimulate blooming. Reduce irrigation pre-flowering, then increase after fruit set.",
    "Grapes"      : "Grapes require careful irrigation management. Reduce water before harvest to concentrate sugars — drip irrigation is recommended.",
    "Watermelon"  : "Watermelon needs regular watering especially during fruit development. Reduce irrigation as fruit matures to improve sweetness.",
    "Muskmelon"   : "Muskmelon needs consistent moisture during growth but reduce irrigation near harvest for better flavour and sugar content.",
    "Apple"       : "Apple trees need deep, infrequent watering. Ensure good drainage — waterlogging causes root diseases.",
    "Orange"      : "Oranges need regular moisture but are sensitive to waterlogging. Irrigate deeply and allow soil to dry slightly between waterings.",
    "Papaya"      : "Papaya is sensitive to both drought and waterlogging. Maintain moist but well-drained soil and irrigate frequently in dry weather.",
    "Coconut"     : "Coconut palms need regular watering, especially in dry seasons. Basin irrigation works well — maintain moisture around the root zone.",
    "Cotton"      : "Cotton needs moderate water. Critical stages are squaring and boll development — avoid water stress during these periods.",
    "Jute"        : "Jute requires high moisture and humidity. Irrigate regularly and ensure the field is well-watered throughout the growing season.",
    "Coffee"      : "Coffee needs consistent moisture but good drainage. Irrigate during dry spells — water stress during flowering reduces berry yield."
}

# ============================================================
#  IRRIGATION AMOUNT LOGIC
#  Formula: base amount adjusted by temperature, humidity,
#  rainfall already received, and crop water demand
# ============================================================

CROP_BASE_MM = {
    "Rice"        : 50,
    "Maize"       : 35,
    "Chickpea"    : 20,
    "Kidney Beans": 30,
    "Pigeon Peas" : 18,
    "Moth Beans"  : 15,
    "Mung Bean"   : 25,
    "Black Gram"  : 25,
    "Lentil"      : 20,
    "Pomegranate" : 22,
    "Banana"      : 45,
    "Mango"       : 28,
    "Grapes"      : 25,
    "Watermelon"  : 35,
    "Muskmelon"   : 30,
    "Apple"       : 28,
    "Orange"      : 30,
    "Papaya"      : 35,
    "Coconut"     : 40,
    "Cotton"      : 32,
    "Jute"        : 40,
    "Coffee"      : 28
}

def calculate_irrigation_amount(crop, temperature, humidity, rainfall):
    base = CROP_BASE_MM.get(crop, 30)

    # Increase amount if temperature is very high
    if temperature > 35:
        base += 10
    elif temperature > 30:
        base += 5

    # Increase amount if humidity is very low
    if humidity < 30:
        base += 8
    elif humidity < 50:
        base += 4

    # Reduce amount based on recent rainfall already received
    rainfall_offset = min(rainfall * 0.2, 15)
    base -= rainfall_offset

    # Clamp to sensible range
    base = max(10, min(base, 70))

    return round(base)

CROP_LIST = list(CROP_ADVICE.keys())

# ============================================================
#  HEADER
# ============================================================

st.markdown("## 🌱 Crop Irrigation Need Predictor")
st.markdown("Select your crop and enter farm conditions to find out if irrigation is needed today.")
st.markdown("---")

# ============================================================
#  SDG BADGES
# ============================================================

col_s1, col_s2, col_s3 = st.columns([1, 1, 4])
with col_s1:
    st.success("🌾 SDG 2 — Zero Hunger")
with col_s2:
    st.success("🌍 SDG 15 — Life on Land")

st.markdown("---")

# ============================================================
#  CROP SELECTOR
# ============================================================

st.markdown("### Select Your Crop")
selected_crop = st.selectbox(
    "Choose the crop you are growing",
    options=CROP_LIST,
    index=0,
    help="Select your crop type — you will receive crop-specific irrigation advice after prediction."
)

st.markdown("---")

# ============================================================
#  INPUT SLIDERS — TWO COLUMNS
# ============================================================

st.markdown("### Farm Conditions")

col1, col2 = st.columns(2)

with col1:
    N = st.slider(
        "Nitrogen (N) — soil nutrient",
        min_value=0, max_value=140,
        value=50, step=1,
        help="Amount of Nitrogen in the soil (kg/ha)"
    )
    P = st.slider(
        "Phosphorus (P) — soil nutrient",
        min_value=5, max_value=145,
        value=50, step=1,
        help="Amount of Phosphorus in the soil (kg/ha)"
    )
    K = st.slider(
        "Potassium (K) — soil nutrient",
        min_value=5, max_value=205,
        value=50, step=1,
        help="Amount of Potassium in the soil (kg/ha)"
    )
    temperature = st.slider(
        "Temperature (°C)",
        min_value=5.0, max_value=45.0,
        value=25.0, step=0.5,
        help="Average daily temperature in Celsius"
    )

with col2:
    humidity = st.slider(
        "Humidity (%)",
        min_value=10.0, max_value=100.0,
        value=70.0, step=0.5,
        help="Relative humidity percentage"
    )
    ph = st.slider(
        "Soil pH",
        min_value=3.5, max_value=10.0,
        value=6.5, step=0.1,
        help="pH level of the soil (3.5 = acidic, 10 = alkaline)"
    )
    rainfall = st.slider(
        "Rainfall (mm)",
        min_value=20.0, max_value=300.0,
        value=100.0, step=1.0,
        help="Amount of rainfall received in mm"
    )

st.markdown("---")

# ============================================================
#  PREDICT BUTTON
# ============================================================

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_clicked = st.button(
        "🔍 Predict Irrigation Need",
        use_container_width=True
    )

# ============================================================
#  PREDICTION OUTPUT
# ============================================================

if predict_clicked:

    input_data = pd.DataFrame([{
        "N"          : N,
        "P"          : P,
        "K"          : K,
        "temperature": temperature,
        "humidity"   : humidity,
        "ph"         : ph,
        "rainfall"   : rainfall
    }])

    input_scaled    = scaler.transform(input_data)
    prediction      = model.predict(input_scaled)[0]
    probability     = model.predict_proba(input_scaled)[0]
    confidence      = round(max(probability) * 100, 2)
    result_label    = "Irrigate" if prediction == 1 else "No Irrigation"

    # Calculate irrigation amount only if needed
    irrigation_mm   = calculate_irrigation_amount(
        selected_crop, temperature, humidity, rainfall
    ) if prediction == 1 else 0

    st.markdown("---")
    st.markdown(f"### Prediction Result for: {selected_crop}")

    if prediction == 1:
        st.error("## 🚨 Irrigation NEEDED")
        st.markdown(
            "Your crop is likely **water-stressed** based on current "
            "soil and weather conditions. Consider irrigating soon to "
            "avoid yield loss."
        )

        # Irrigation amount suggestion
        amt_col1, amt_col2, amt_col3 = st.columns([1, 2, 1])
        with amt_col2:
            st.warning(
                f"### 💧 Suggested Irrigation Amount: **{irrigation_mm} mm**\n\n"
                f"Based on your crop type ({selected_crop}), current temperature "
                f"({temperature}°C), humidity ({humidity}%), and recent rainfall "
                f"({rainfall} mm)."
            )

    else:
        st.success("## ✅ No Irrigation Needed")
        st.markdown(
            "Your soil has **sufficient moisture** based on current "
            "conditions. No irrigation is required right now. Check "
            "again tomorrow or after weather changes."
        )

    # Confidence bar
    st.markdown(f"**Model Confidence: {confidence}%**")
    st.progress(int(confidence))

    # Crop-specific advice
    st.markdown("---")
    st.markdown(f"### 🌿 Irrigation Advice for {selected_crop}")
    st.info(CROP_ADVICE[selected_crop])

    # Input summary
    st.markdown("---")
    st.markdown("### Input Summary")
    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.metric("Crop",           selected_crop)
        st.metric("Nitrogen (N)",   f"{N} kg/ha")
        st.metric("Phosphorus (P)", f"{P} kg/ha")
        st.metric("Potassium (K)",  f"{K} kg/ha")

    with summary_col2:
        st.metric("Temperature",    f"{temperature} °C")
        st.metric("Humidity",       f"{humidity} %")
        st.metric("Rainfall",       f"{rainfall} mm")
        st.metric("Prediction",     result_label)

    # ============================================================
    #  SAVE TO HISTORY
    # ============================================================

    st.session_state.history.append({
        "Crop"          : selected_crop,
        "Temp (°C)"     : temperature,
        "Humidity (%)"  : humidity,
        "Rainfall (mm)" : rainfall,
        "N"             : N,
        "P"             : P,
        "K"             : K,
        "pH"            : ph,
        "Prediction"    : result_label,
        "Amount (mm)"   : irrigation_mm if prediction == 1 else "—",
        "Confidence (%)" : confidence
    })

# ============================================================
#  RESULT HISTORY TABLE
# ============================================================

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📋 Prediction History — This Session")

    history_df = pd.DataFrame(st.session_state.history)
    history_df.index = range(1, len(history_df) + 1)
    history_df.index.name = "#"

    st.dataframe(history_df, use_container_width=True)

    # Clear history button
    clr_col1, clr_col2, clr_col3 = st.columns([1, 2, 1])
    with clr_col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ============================================================
#  FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:13px;'>"
    "Crop Irrigation Need Predictor · Built with Streamlit & scikit-learn · "
    "Supporting SDG 2 and SDG 15"
    "</div>",
    unsafe_allow_html=True
)
