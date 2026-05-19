import streamlit as st
import pandas as pd
import pickle

# 1. Page Configuration
st.set_page_config(page_title="Iris Species Predictor", layout="centered")
st.title("Iris Flower Classification By Sir Rehan ")
st.markdown("### Predict the species of an Iris flower based on its structural measurements.")

# 2. Load the trained Iris model
@st.cache_resource
def load_model():
    # Make sure this pkl file contains your trained Iris model!
    with open("best_model.pkl", "rb") as f:
        model_info = pickle.load(f)
    
    # Handles both direct models and custom dictionaries containing metadata
    if isinstance(model_info, dict):
        return model_info["model"], model_info.get("classes")
    return model_info, None

model, classes = load_model()

# 3. Sidebar for User Inputs (Iris Features)
st.sidebar.header("Enter Flower Measurements (cm)")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.8, step=0.1)
    sepal_width  = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0, step=0.1)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 3.8, step=0.1)
    petal_width  = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.2, step=0.1)
    
    # Create dictionary matching the feature layout your model expects
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Optional: Function to save the backend prediction data into XAMPP
def save_to_xampp(df, prediction_label):
    try:
        import mysql.connector
        # Establishes connection to your XAMPP MySQL setup
        conn = mysql.connector.connect(
            host="localhost",
            user="root",        # XAMPP default username
            password="",        # XAMPP default password (empty)
            database="mlops"    # Your database name from the screenshot
        )
        cursor = conn.cursor()
        
        # SQL structure targeting your 'iris' table
        query = """
        INSERT INTO iris (sepal_length, sepal_width, petal_length, petal_width, prediction) 
        VALUES (%s, %s, %s, %s, %s)
        """
        values = (
            float(df['sepal_length'].iloc[0]), 
            float(df['sepal_width'].iloc[0]), 
            float(df['sepal_length'].iloc[0]), 
            float(df['sepal_width'].iloc[0]), 
            str(prediction_label)
        )
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as db_err:
        st.warning(f"Database logging skipped: {db_err}")

# 4. Prediction Logic
if st.button("🔍 Predict Iris Species", type="primary"):
    with st.spinner("Classifying features..."):
        try:
            # Predict index class
            prediction = model.predict(input_df)[0]
            
            # Map predictions to species strings
            # If your model metadata doesn't contain explicit string classes, fallback to standard dataset indexing
            if classes is not None:
                result = classes[prediction]
            else:
                iris_target_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
                result = iris_target_names.get(prediction, f"Unknown (Class {prediction})")

            # Display Results
            st.success(f"**Predicted Species: {str(result).upper()}**")
            
            # Display prediction probabilities if the model supports it
            try:
                probability = model.predict_proba(input_df)[0]
                st.markdown("#### Prediction Confidence:")
                
                # Check if multi-class probabilities match standard iris limits
                if len(probability) >= 3:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Setosa", f"{probability[0]*100:.1f}%")
                    with c2:
                        st.metric("Versicolor", f"{probability[1]*100:.1f}%")
                    with c3:
                        st.metric("Virginica", f"{probability[2]*100:.1f}%")
            except:
                pass # Model variant doesn't support probability estimation
            
            st.balloons()
            
            # --- UNCOMMENT THE LINE BELOW TO ENABLE AUTOMATIC XAMPP BACKEND SAVING ---
            # save_to_xampp(input_df, result)
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# 5. Model Diagnostics Info
try:
    model_type = type(model.named_steps['model']).__name__
    st.info(f"Model Engine: **{model_type}**")
except:
    st.info("Iris Classification Model loaded successfully.")