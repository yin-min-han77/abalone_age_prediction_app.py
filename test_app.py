import streamlit as st
import pandas as pd
import pickle

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Abalone Age Predictor",
    page_icon="🦪",
    layout="centered"
)

# --- 2. THEME & BACKGROUND IMAGE SETUP ---
# We use a public URL for an underwater image.
# We also force the text color to be dark so it shows up on the light glass panels.
page_bg_img = """
<style>
/* 1. Main Background Image */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1583212292454-1fe6229603b7?q=80&w=1974&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* 2. Sidebar Background - Semi-transparent Deep Blue */
[data-testid="stSidebar"] {
    background-color: rgba(0, 60, 80, 0.95); 
}

/* 3. Make the main Header text white so it pops against the water */
h1 {
    color: #FFFFFF !important;
    text-shadow: 2px 2px 4px #000000;
}

/* 4. Create a "Glass" effect for the containers */
div.block-container {
    background-color: rgba(255, 255, 255, 0.85); /* White with 85% opacity */
    padding: 2rem;
    border-radius: 15px;
    margin-top: 2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

/* 5. Force text inside the white box to be black/dark teal */
p, label, h2, h3, .stMarkdown {
    color: #004D40 !important;
}

/* 6. Style the Buttons */
.stButton>button {
    background-color: #FF8A65; /* Coral color */
    color: white !important;
    border: none;
    border-radius: 8px;
    font-size: 18px;
}
.stButton>button:hover {
    background-color: #FF5722;
    color: white !important;
}

/* 7. Input fields styling */
.stNumberInput input, .stSelectbox div[data-baseweb="select"] div {
    background-color: #E0F7FA; /* Light blue input background */
    color: #006064;
}

/* 8. Sidebar Text Color (Must be white to see against dark blue sidebar) */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
    color: #E0F7FA !important;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# --- 3. Sidebar Content ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2821/2821876.png", width=80)
    st.header("Student Information")
    st.write("Name: **Yin Min Han**")
    st.write("ID: **PIUS20230069**")
    st.markdown("---")
    st.write("Model: **Linear Regression**")
    st.info("🌊 **Abalone Fact:** They cling to rocks with exceptional strength!")


# --- 4. Main App Content ---
st.title("🦪 Abalone Age Estimator")
st.write("Adjust the inputs below to match the abalone's measurements.")

# --- Load Resources ---
@st.cache_resource
def load_data():
    try:
        with open('abalone_model.pkl', 'rb') as file:
            model, encoder = pickle.load(file)
        return model, encoder
    except FileNotFoundError:
        return None, None

model, encoder = load_data()

if model is None:
    st.error("⚠️ Error: 'abalone_model.pkl' not found.")
else:
    # --- Step 1: Gender ---
    st.markdown("### Step 1: Gender")
    sex = st.selectbox("Select Sex", options=['M', 'F', 'I'], 
                       format_func=lambda x: "Male" if x == 'M' else ("Female" if x == 'F' else "Infant"))

    # --- Step 2: Dimensions ---
    st.markdown("### Step 2: Size Measurements (mm)")
    c1, c2, c3 = st.columns(3)
    with c1:
        length = st.number_input("Length", value=0.455, step=0.01)
    with c2:
        diameter = st.number_input("Diameter", value=0.365, step=0.01)
    with c3:
        height = st.number_input("Height", value=0.095, step=0.01)

    # --- Step 3: Weights ---
    st.markdown("### Step 3: Weight Measurements (grams)")
    c4, c5 = st.columns(2)
    with c4:
        whole_weight = st.number_input("Whole Weight", value=0.514, step=0.01)
        shucked_weight = st.number_input("Shucked Weight", value=0.224, step=0.01)
    with c5:
        viscera_weight = st.number_input("Viscera Weight", value=0.101, step=0.01)
        shell_weight = st.number_input("Shell Weight", value=0.150, step=0.01)

    st.markdown("---")

    # --- Prediction ---
    if st.button("Predict Age 🌊"):
        # Create DataFrame
        input_data = pd.DataFrame({
            'Sex': [sex],
            'Length': [length],
            'Diameter': [diameter],
            'Height': [height],
            'Whole_weight': [whole_weight],
            'Shucked_weight': [shucked_weight],
            'Viscera_weight': [viscera_weight],
            'Shell_weight': [shell_weight]
        })

        # Encode
        input_data[['Sex']] = encoder.transform(input_data[['Sex']])

        # Predicta
        prediction = model.predict(input_data)[0]
        
        # Result Display
        st.success(f"### 🐚 Predicted Rings: {prediction:.2f}")
        st.info(f"**Estimated Age:** approx. {prediction + 1.5:.1f} years old")