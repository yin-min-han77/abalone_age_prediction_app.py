import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Abalone Age Predictor")

st.title("🦪 Abalone Age Prediction App")
st.write("Enter the abalone measurements below to predict its age.")

# Sidebar
with st.sidebar:
    st.header("Student Information")
    st.write("Name: **Yin Min Han**")
    st.write("ID: **PIUS20230069**")
    st.write("---")
    st.write("Model: Linear Regression")

# Load the Model and Encoder ---
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
    st.error("Error: 'abalone_model.pkl' not found. Please run your training script first.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dimensions")
        sex = st.selectbox("Sex", options=['M', 'F', 'I'], format_func=lambda x: "Male" if x == 'M' else ("Female" if x == 'F' else "Infant"))
        length = st.number_input("Length (mm)", value=0.455)
        diameter = st.number_input("Diameter (mm)", value=0.365)
        height = st.number_input("Height (mm)", value=0.095)

    with col2:
        st.subheader("Weights")
        whole_weight = st.number_input("Whole Weight (g)", value=0.514)
        shucked_weight = st.number_input("Shucked Weight (g)", value=0.224)
        viscera_weight = st.number_input("Viscera Weight (g)", value=0.101)
        shell_weight = st.number_input("Shell Weight (g)", value=0.150)

    if st.button("Predict Age"):
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

        # Encode 'Sex' using the saved encoder
        input_data[['Sex']] = encoder.transform(input_data[['Sex']])

        # Predict
        prediction = model.predict(input_data)
        rings = prediction[0]

        # Display Result
        st.success(f"Predicted Rings: {rings:.2f}")
        st.info(f"Estimated Age: {rings + 1.5:.1f} years")