from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import requests
from io import BytesIO
import os
import time
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Custom styles to change white text to black and style sidebar buttons
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E3F2FD; /* Light pastel blue */
    }
    
    /* Sidebar links styling */
    .sidebar-link {
        font-size: 18px;
        font-weight: bold;
        padding: 15px;
        border-radius: 5px;
        color: #2E9AFF; /* Soft purple */
        text-decoration: none;
        display: block;
        margin: 10px 0;
        transition: color 0.3s ease, background-color 0.3s ease;
    }
    .sidebar-link:hover {
        color: white;
        text-decoration: none;
        background-color: rgba(255,255,255,0.5);
    }

    /* Style for sidebar radio buttons */
    .st-eb {
        font-size: 18px !important;
        font-weight: bold !important;
        color: #2E9AFF !important;
        padding: 10px !important;
        border-radius: 5px !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        transition: background-color 0.3s ease, color 0.3s ease !important;
    }
    .st-eb:hover {
        background-color: rgba(255, 255, 255, 0.3) !important;
        color: white !important;
    }
    .st-eb:checked {
        background-color: #2E9AFF !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Google Drive file ID
drive_file_id = "1os3m_b2PYcvCz33Ku_vzRjSkhBh8Y7e5"
model_path = "resnet_vit_model.h5"

# Function to download model
def download_model():
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... Please wait."):
            gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", model_path, quiet=False)

# Download model if not exists
download_model()

# Load model
if "model_loaded" not in st.session_state:
    try:
        with st.spinner("Loading model..."):
            model = tf.keras.models.load_model(model_path, compile=False)
        st.session_state["model_loaded"] = True
        st.session_state["model"] = model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.session_state["model"] = None
else:
    model = st.session_state["model"]

# Define class labels and air quality descriptions
class_names = [
    "a_Good", "b_Moderate", "c_Unhealthy_for_Sensitive_Groups",
    "d_Unhealthy", "e_Very_Unhealthy", "f_Severe"
]

aq_descriptions = {
    "a_Good": {
        "Label": "Good",
        "Precaution": [
            "Enjoy outdoor activities without restrictions.",
            "Keep indoor spaces ventilated with fresh air.",
            "Maintain greenery and plant trees to support clean air.",
            "Reduce pollution by using eco-friendly transportation."
        ]
    },
    "b_Moderate": {
        "Label": "Moderate",
        "Precaution": [
            "Sensitive individuals (children, elderly, and respiratory patients) should limit prolonged outdoor activities.",
            "Avoid intense outdoor exercise during morning (6-10 AM) and evening (5-9 PM) when pollution is higher due to traffic and industrial emissions.",
            "If you experience discomfort (coughing, breathing issues, or irritation), consider staying indoors.",
            "Keep indoor air fresh by ensuring proper ventilation and avoiding smoke indoors.",
            "Reduce personal pollution by using public transport, carpooling, or cycling instead of private vehicles."
        ]
    },
    "c_Unhealthy_for_Sensitive_Groups": {
        "Label": "Unhealthy for Sensitive Groups",
        "Precaution": [
            "Wear a light mask if exposure is prolonged.",
            "Keep windows closed to prevent outdoor pollutants from entering.",
            "Use air purifiers or ventilation filters indoors if air quality feels irritating.",
            "Keep windows closed during peak pollution hours (6-10 AM, 5-9 PM).",
            "Limit activities that create indoor pollution, such as burning candles or smoking."
        ]
    },
    "d_Unhealthy": {
        "Label": "Unhealthy",
        "Precaution": [
            "Everyone should limit prolonged outdoor exposure.",
            "Outdoor exercise is not recommended.",
            "Wear an N95 mask if going outside.",
            "Keep windows/doors closed & use an air purifier indoors.",
            "Avoid activities that worsen indoor pollution (candles, smoking, strong cooking fumes).",
            "Wash hands and face after being outside to remove pollutants."
        ]
    },
    "e_Very_Unhealthy": {
        "Label": "Very Unhealthy",
        "Precaution": [
            "Everyone should stay indoors unless absolutely necessary.",
            "Everyone should avoid outdoor activities; only go outside if absolutely necessary.",
            "Sensitive individuals (children, elderly, respiratory/heart patients) must stay indoors and monitor symptoms.",
            "Wear an N95 or higher-grade mask when stepping outside.",
            "Keep windows and doors shut; use air purifiers indoors.",
            "Avoid activities that worsen indoor air (smoking, frying, candles).",
            "Drink plenty of water to stay hydrated and reduce throat/lung irritation."
        ]
    },
    "f_Severe": {
        "Label": "Hazardous",
        "Precaution": [
            "All outdoor activities should be avoided, even for short periods.",
            "N95 masks are mandatory if stepping out in extreme situations.",
            "Use multiple air purifiers if available and seal gaps in windows/doors.",
            "Avoid all indoor pollution sources, including incense, smoking, and gas stoves.",
            "Sensitive individuals should seek immediate medical help if experiencing breathing issues, chest pain, or dizziness.",
            "Steam inhalation and humidifiers can help ease respiratory discomfort."
        ]
    }
}

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Sidebar navigation with styled radio buttons
page = st.sidebar.radio(
    "Navigation",
    ["Home", "About Us", "Model Visualisations"],
    key="navigation"
)

# Page content
if page == "Home":
    st.markdown(
    """
    <h3 style="color: black;font-weight:bold;">
        🌍 An Optimized Convolutional Neural Network Framework for Precise Multi-Class Classification and Comprehensive Analysis of the Air Quality Index to Support Sustainable Urban Development and Monitor Environmental Impacts
    </h3>
    <h2 style="color: #2E9AFF;font-size:22px">
        📌 Upload an image to classify the air quality level.
    </h2>
    """,
    unsafe_allow_html=True
    )
    
    # Show model success message only on Home page
    if "success_message_shown" not in st.session_state:
        success_placeholder = st.empty()
        success_placeholder.markdown(
            """
            <div style="
                background-color:rgba(14, 215, 81, 0.5); 
                color: black; 
                padding: 10px; 
                border-radius: 5px; 
                font-size: 16px; 
                font-weight: bold;
            ">
                ✅ Model successfully loaded!
            </div>
            """, 
            unsafe_allow_html=True
        )
        time.sleep(3)  # Show for 3 seconds
        success_placeholder.empty()
        st.session_state["success_message_shown"] = True  # Prevent further display

    uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "png"])

    # Image preprocessing function
    def preprocess_image(image):
        """Preprocess the image for prediction."""
        image = img_to_array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image  # Shape: (1, 224, 224, 3)

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("🚀 Classify"):
            if model is None:
                st.error("⚠ Model is not loaded. Please check for errors.")
            else:
                try:
                    # Load and preprocess image
                    image = load_img(uploaded_file, target_size=(224, 224))
                    image_preprocessed = preprocess_image(image)

                    # Make prediction
                    predictions = model.predict(image_preprocessed)
                    predicted_index = np.argmax(predictions)
                    predicted_class = class_names[predicted_index]

                    # Get AQI details
                    result = aq_descriptions[predicted_class]

                    # Display results
                    st.markdown(
                    f"""
                    <h2 style="color: black;font-size:22px">
                        Prediction: {result['Label']}
                    </h2>
                    <hr style="background-color:black;"/>
                    """,
                    unsafe_allow_html=True
                    )
                    precaution_html = "<ul style='list-style-type: disc;'>"
                    for p in result["Precaution"]:
                        precaution_html += f"<li style='color:#2E9AFF;font-size:22px;'>{p}</li>"
                    precaution_html += "</ul>"

                    # Show warning for hazardous air quality
                    if predicted_class in ["e_Very_Unhealthy", "f_Severe"]:
                        st.markdown(
                            '<div style="padding: 10px; background-color: #ffcccc; color: red; border-radius: 5px; font-weight: bold;">'
                            '🚨 High pollution levels detected! Limit outdoor exposure.'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    st.markdown(
                    f"""
                    <h2 style="color: black;font-size:22px">
                        <u>Precautions</u>
                    </h2>
                    {precaution_html}
                    """,
                    unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")

elif page == "About Us":
    st.markdown(
        """
        <h3 style="color: black;font-weight:bold;">
            📊 About Us
        </h3>
        <hr style="background-color:black;"/>
        <p style="line-height: 1.5; color: #2E9AFF;font-size:20px;">
            Our project leverages advanced deep learning techniques to analyze environmental images and classify air quality into six distinct categories. 
            By utilizing a hybrid model combining Vision Transformers (ViT) and ResNet architectures, we achieve high accuracy in detecting pollution levels 
            and provide actionable insights to support sustainable urban development.
        </p>
        <p style="line-height: 1.5; color: #2E9AFF;font-size:20px;">
            The system is designed to monitor air quality trends, raise public awareness, and offer precautionary measures to minimize health risks. 
            By integrating artificial intelligence into environmental monitoring, our project aims to contribute to a cleaner, healthier, and more sustainable future.
        </p>
        <p style="line-height: 1.5; color: #2E9AFF;font-size:20px;">
            Key Features:
            <ul style="list-style-type: disc; color: #2E9AFF;font-size:20px;">
                <li>Accurate air quality classification using ViT + ResNet.</li>
                <li>Real-time monitoring and trend analysis.</li>
                <li>Precautionary measures tailored to each air quality level.</li>
                <li>Support for sustainable urban planning and policy-making.</li>
            </ul>
        </p>
        """,
        unsafe_allow_html=True
    )
    
elif page == "Model Visualisations":
    st.markdown(
        """
        <h3 style="color: black;font-weight:bold;">
            📈 Model Visualisations
        </h3>
        <h2 style="color: #2E9AFF;font-size:22px">
            Graphs and charts related to air quality classification
        </h2>
        <hr style="background-color:black;"/>
        <p style="line-height: 1.5; color: #2E9AFF;font-size:20px;">
            The ViT + ResNet model demonstrated superior performance compared to other models, as evident from the visualizations below. 
            It achieved higher accuracy, lower loss, and balanced precision-recall scores, indicating effective feature extraction and classification. 
            The training and validation trends suggest minimal overfitting, confirming the model's robustness in handling air pollution classification tasks.
        </p>
        <br/>
        """,
        unsafe_allow_html=True
    )

    # Function to convert Google Drive view links to direct download links
    def get_direct_link(drive_link):
        file_id = drive_link.split("/")[-2]  # Extract the file ID
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    # Google Drive image links
    image_links = [
        "https://drive.google.com/file/d/1H3v3pEax0nJrqDA5ebG8FxRCFwiWdzRO/view",
        "https://drive.google.com/file/d/1cQVmSUY1IsxmQ5Bqzvp2_jUUvuVN7X8G/view",
        "https://drive.google.com/file/d/1aoD6z5i_0gz4jcrEmXBqYQCqMrEg_xxe/view",
        "https://drive.google.com/file/d/1YUaUTp4Evsqc6KFoqjhRFQ-tqG0xVSKX/view",
        "https://drive.google.com/file/d/1CNafEp_bJIAC3dWZo3ezH7Nof8VYOzOq/view",
        "https://drive.google.com/file/d/1VzW27YS7bPnSOYhsUznWZdkcOUCTTr6E/view",
        "https://drive.google.com/file/d/1tYvUD09jBp094PxYFp2g70xVQ6llo8L-/view",
        "https://drive.google.com/file/d/1B9x_Wli8AoP-gViKSbwl9la--fWrQwiy/view",
        "https://drive.google.com/file/d/1MvAh-_TSuupJLwrU7QnKW2tHUR8U-V5p/view",
        "https://drive.google.com/file/d/1VizyB6naeQMUQD11duf43oc4-BzDpaQp/view",
        "https://drive.google.com/file/d/1Ok3zO9TnDK3pGaVi6QvoPqhsKS5JAmlX/view",
        "https://drive.google.com/file/d/1fy4wgjuH3c0SPAz3efWE5hjufZAvYAPw/view",
        "https://drive.google.com/file/d/1vmsEbMOHuC8jthMYNAKWfzF9qTnkDKFR/view"
    ]

    # Display images
    for idx, link in enumerate(image_links):
        direct_link = get_direct_link(link)
        
        try:
            response = requests.get(direct_link, stream=True)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.image(image, caption=f"Visualization {idx + 1}", use_container_width=True)
            else:
                st.warning(f"⚠ Could not load Visualization {idx + 1}")
        
        except Exception as e:
            st.error(f"Error loading Visualization {idx + 1}: {e}")