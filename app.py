import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import plotly.io as pio
import os

# Load model and preprocessors
model = joblib.load('best_model_decision_tree.pkl')
gender_encoder = joblib.load('Gender_label_encoder.pkl')
occupation_encoder = joblib.load('Occupation_label_encoder.pkl')
bmi_encoder = joblib.load('BMI Category_label_encoder.pkl')
scaler = joblib.load('minmax_scaler_split.pkl')

# Load all Plotly figures from the "templates" folder
def load_plotly_figures(folder_path):
    html_files = [f for f in os.listdir(folder_path) if f.endswith('.html')]
    html_contents = {}
    for html_file in html_files:
        with open(os.path.join(folder_path, html_file), "r", encoding="utf-8") as f:
            html_contents[html_file] = f.read()
    return html_contents

def set_bg_hack_url():

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://media.istockphoto.com/id/1354171846/photo/modern-medical-research-laboratory-two-scientists-wearing-face-masks-use-microscope-analyse.jpg?s=612x612&w=0&k=20&c=SEVWRjqmrTgy2axQBGv946130gsUJZcEUfo-fLqT2M4=");
            background-size: cover;
            backdrop-filter: blur(10px);
        }}
        .main {{
            background: rgba(0, 0, 0, 0.7);
            border-radius: 10px;
            padding: 10px;
        }}
        .css-18e3th9 {{
            padding-top: 2rem;
        }}
        .css-1d391kg {{
            backdrop-filter: blur(10px);
            background: rgba(255, 255, 255, 0.1);
        }}
        .css-12oz5g7 {{
            color: white;
        }}
        .css-1n76uvr {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Modify the predict_sleep_disorder function to accept a list of features
def predict_sleep_disorder(features):
    try:
        # Make predictions
        prediction = model.predict(features.reshape(1, -1))
        # Return the predicted sleep disorder
        return prediction[0]
    except ValueError as e:
        return str(e)

def main():
    set_bg_hack_url()

    # Create sidebar menu
    st.sidebar.title("Menu")
    menu_option = st.sidebar.radio("Go to", ("🌟 Prediction", "📊 Analytics"))

    st.markdown('<div class="main">', unsafe_allow_html=True)

    if menu_option == "🌟 Prediction":
        st.title("Comprehensive Sleep Disorder Prediction and Analysis System")
        st.write("This App Identifies Sleep Disorders Using Input Features 🛌")

        # Create columns for input features
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("📝 Input Details")
            # Age
            age = st.slider("Age", 18, 100, 27)
            st.markdown("**Age**: The individual's age in years.")
            
            # Gender
            gender = st.selectbox("Gender", ["Male", "Female"])
            st.markdown("**Gender**: The individual's gender.")
            
            # Occupation
            occupation = st.selectbox("Occupation", [
                'Doctor', 'Teacher', 'Nurse', 'Engineer', 'Accountant', 
                'Lawyer', 'Salesperson', 'Others'
            ])
            st.markdown("**Occupation**: The individual's occupation.")

            # BMI Category
            bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
            st.markdown("**BMI Category**: The individual's BMI classification.")

        with col2:
            st.header("💤 Sleep Details")
            # Sleep Duration
            sleep_duration = st.slider("Sleep Duration", 0.0, 24.0, 4.0, 0.1)
            st.markdown("**Sleep Duration**: The number of hours the individual sleeps each night.")
            
            # Quality of Sleep
            quality_of_sleep = st.slider("Quality of Sleep", 0, 10, 5)
            st.markdown("**Quality of Sleep**: A self-assessed rating of sleep quality on a scale from 0 to 10.")
            
            # Physical Activity Level
            physical_activity_level = st.slider("Physical Activity Level", 0, 100, 50)
            st.markdown(
                """
                **Physical Activity Level**: 
                - **0-33**: Low (Little or no physical activity)
                - **34-66**: Moderate (Some physical activity such as walking, light exercise)
                - **67-100**: High (Regular or intense physical activity such as gym workouts, running)
                """
            )

        with col3:
            st.header("💓 Health Details")
            # Stress Level
            stress_level = st.slider("Stress Level", 0, 10, 5)
            st.markdown(
                """
                **Stress Level**: 
                - **0**: No stress
                - **1-3**: Low stress
                - **4-6**: Moderate stress
                - **7-9**: High stress
                - **10**: Extremely high stress
                """
            )
            
            # Heart Rate
            heart_rate = st.number_input("Heart Rate", value=70, step=1)
            st.markdown("**Heart Rate**: The individual's resting heart rate in beats per minute.")
            
            # Daily Steps
            daily_steps = st.number_input("Daily Steps", value=5000, step=1000)
            st.markdown("**Daily Steps**: The average number of steps the individual takes daily.")
            
            # Systolic
            systolic = st.number_input("Systolic", value=120, step=1)
            st.markdown("**Systolic**: The individual's systolic blood pressure in mmHg.")
            
            # Diastolic
            diastolic = st.number_input("Diastolic", value=80, step=1)
            st.markdown("**Diastolic**: The individual's diastolic blood pressure in mmHg.")

        # Encode categorical variables using loaded LabelEncoders
        gender_num = gender_encoder.transform([gender])[0]
        occupation_num = occupation_encoder.transform([occupation])[0]
        bmi_category_num = bmi_encoder.transform([bmi_category])[0]

        # List of features for scaling
        numerical_features = [
            age, sleep_duration, quality_of_sleep, physical_activity_level, 
            stress_level, heart_rate, daily_steps, systolic, diastolic
        ]

        # Scale numerical features using loaded MinMaxScaler
        # Ensure the input has the same number of features the scaler was fitted on
        complete_features = np.zeros((1, 12))
        complete_features[0, :9] = numerical_features  # Assuming the first 9 features are numerical

        scaled_features = scaler.transform(complete_features).flatten()

        # Features list
        features = np.array([
            gender_num,
            scaled_features[0],  # age_scaled
            occupation_num,
            scaled_features[1],  # sleep_duration_scaled
            scaled_features[2],  # quality_of_sleep_scaled
            scaled_features[3],  # physical_activity_level_scaled
            scaled_features[4],  # stress_level_scaled
            bmi_category_num,
            scaled_features[5],  # heart_rate_scaled
            scaled_features[6],  # daily_steps_scaled
            scaled_features[7],  # systolic_scaled
            scaled_features[8]   # diastolic_scaled
        ])



        # Make prediction
        if st.button("🔮 Predict"):
            prediction = predict_sleep_disorder(features)
            if prediction == 0:
                st.success("No Sleep Disorder Detected")
                st.markdown("""
                    **Tips for Maintaining Healthy Sleep:**
                    - Maintain a consistent sleep schedule, even on weekends. 🕒
                    - Engage in a relaxing bedtime routine. 🌙
                    - Avoid napping, especially in the afternoon. 🛏️
                    - Exercise regularly. 🏃‍♂️
                    - Ensure your bedroom is comfortable in terms of temperature, noise, and lighting. 🛌
                """)
            elif prediction == 2:
                st.warning("Sleep Disorder Detected: Sleep Apnea")
                st.markdown("""
                    **Tips for Managing Sleep Apnea:**
                    - Maintain a healthy weight. 🏋️
                    - Sleep on your side if possible. 😴
                    - Avoid alcohol and smoking. 🚫
                    - Use a CPAP device as prescribed. 🦺
                    - Follow good sleep hygiene practices. 🛏️
                """)
            elif prediction == 1:
                st.error("Sleep Disorder Detected: Insomnia")
                st.markdown("""
                    **Tips for Managing Insomnia:**
                    - Keep a regular sleep schedule. 📅
                    - Avoid caffeine and nicotine close to bedtime. 🚫
                    - Create a comfortable sleeping environment. 🛋️
                    - Manage stress and practice relaxation techniques. 🧘‍♂️
                    - Limit screen time before bed. 📱
                """)


    elif menu_option == "📊 Analytics":
        # Set title and description for Analytics
        st.title("📈 Insights from the Dataset")
        st.write("Explore various analytics and insights derived from the sleep data. 📊")

        # Load and display Plotly figures from the "templates" folder
        html_contents = load_plotly_figures('templates')
        for html_file, html_content in html_contents.items():
            st.markdown(f"### {html_file.replace('.html', '').replace('_', ' ').title()}")
            components.html(html_content, height=700)

    st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
