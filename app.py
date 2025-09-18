import streamlit as st

# Custom CSS for forest background + text + code
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1501785888041-af3ef285b470");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* Make all text white and bold */
html, body, [class*="css"]  {
    color: white;
    font-weight: bold;
    font-family: "Consolas", sans-serif;
}

/* Code blocks (st.code, st.markdown with ```python) */
code, pre {
    color: #00FF00 !important;   /* neon green text */
    background-color: #000000 !important; /* black bg */
    font-weight: bold;
    border-radius: 8px;
    padding: 5px;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.6);
    color: black;
    font-weight: normal;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)




import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


# --- Streamlit app UI ---
st.set_page_config(
    page_title="Plant Disease Diagnosis",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Diagnosis App")
st.markdown("""
### ℹ️ About Plant Disease Diagnosis
Plant disease diagnosis is the process of identifying the type of disease affecting a plant by analyzing **visible symptoms** on its leaves, stems, or fruits.  
Traditionally, this required expert knowledge from farmers or plant pathologists.  

With **AI and Machine Learning**, we can now train models using thousands of plant images to automatically detect:
- 🌿 Whether a plant is healthy or diseased  
- 🦠 The specific disease (fungal, bacterial, viral, or pest-related)  
- 🧪 Possible nutrient deficiencies that mimic disease symptoms  

This app uses a **Convolutional Neural Network (CNN)** trained on plant leaf images to predict the health condition of crops like **Tomato, Potato, and Bell Pepper**.  
It helps farmers and gardeners to:  
- ✅ Get faster disease detection  
- ✅ Reduce crop losses  
- ✅ Take timely action with treatments  
""")

st.write("Upload a leaf image and let the AI diagnose its health status.")

# -------------- Load your model --------------
def load_plant_model():
    model = load_model("plant_disease_model_retrained.3.h5")
    return model

model = load_plant_model()

# -------------- Class names --------------
class_names = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight",
    "Tomato__Target_Spot", "Tomato__Tomato_mosaic_virus", "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_healthy",
    "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]

# -------------- Disease info dict (use your original one here) --------------
disease_info= {
    "Pepper__bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "taxonomy": "Capsicum annuum",
        "status": "Diseased",
        "disease": "Bacterial Spot",
        "cause": "Bacterium Xanthomonas campestris",
        "deficiency": "May resemble magnesium deficiency",
        "diagnosis": "Small, water-soaked spots that enlarge and turn brown with yellow halos"
    },
    "Pepper__bell___healthy": {
        "plant": "Bell Pepper",
        "taxonomy": "Capsicum annuum",
        "status": "Healthy",
        "disease": "None",
        "cause": "N/A",
        "deficiency": "N/A",
        "diagnosis": "No visible disease symptoms"
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "taxonomy": "Solanum tuberosum",
        "status": "Diseased",
        "disease": "Early Blight",
        "cause": "Fungus Alternaria solani",
        "deficiency": "Possible potassium deficiency if severe",
        "diagnosis": "Brown concentric spots on leaves with yellow margins"
    },
    "Potato___healthy": {
        "plant": "Potato",
        "taxonomy": "Solanum tuberosum",
        "status": "Healthy",
        "disease": "None",
        "cause": "N/A",
        "deficiency": "N/A",
        "diagnosis": "No symptoms observed"
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "taxonomy": "Solanum tuberosum",
        "status": "Diseased",
        "disease": "Late Blight",
        "cause": "Oomycete Phytophthora infestans",
        "deficiency": "May resemble calcium deficiency",
        "diagnosis": "Large, irregular brown lesions with white mold underneath leaves"
    },
    "Tomato__Target_Spot": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Target Spot",
        "cause": "Fungus Corynespora cassiicola",
        "deficiency": "May mimic potassium or magnesium deficiency",
        "diagnosis": "Dark, circular spots with concentric rings, especially on older leaves"
    },
    "Tomato__Tomato_mosaic_virus": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Tomato Mosaic Virus",
        "cause": "Tobamovirus",
        "deficiency": "Can be confused with iron deficiency",
        "diagnosis": "Mottled or mosaic light/dark green leaf patterns with distortion"
    },
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "cause": "Begomovirus transmitted by whiteflies",
        "deficiency": "May resemble nitrogen deficiency",
        "diagnosis": "Upward leaf curling, yellowing, stunted growth"
    },
    "Tomato_Bacterial_spot": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Bacterial Spot",
        "cause": "Xanthomonas spp.",
        "deficiency": "May resemble salt stress or toxicity",
        "diagnosis": "Dark, greasy-looking leaf spots, may merge and kill leaves"
    },
    "Tomato_Early_blight": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Early Blight",
        "cause": "Alternaria solani",
        "deficiency": "May mimic magnesium deficiency",
        "diagnosis": "Dark brown spots with concentric rings, lower leaf drop"
    },
    "Tomato_healthy": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Healthy",
        "disease": "None",
        "cause": "N/A",
        "deficiency": "N/A",
        "diagnosis": "No symptoms; normal leaf and stem appearance"
    },
    "Tomato_Late_blight": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Late Blight",
        "cause": "Phytophthora infestans",
        "deficiency": "May resemble bacterial canker in severe stages",
        "diagnosis": "Large, gray-green water-soaked lesions, white mold under humid conditions"
    },
    "Tomato_Leaf_Mold": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Leaf Mold",
        "cause": "Fungus Passalora fulva (Cladosporium)",
        "deficiency": "Can be confused with aging leaves",
        "diagnosis": "Yellow patches on top of leaves and olive-green mold underneath"
    },
    "Tomato_Septoria_leaf_spot": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Septoria Leaf Spot",
        "cause": "Fungus Septoria lycopersici",
        "deficiency": "May be confused with nutrient burn or chemical damage",
        "diagnosis": "Small, circular spots with dark brown margins and gray centers"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Spider Mite Infestation",
        "cause": "Tetranychus urticae (Two-Spotted Spider Mite)",
        "deficiency": "May resemble chlorosis or zinc deficiency",
        "diagnosis": "Stippled yellow leaves with fine webbing, typically on undersides"
    }
}



# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Diagnose"):
        with st.spinner('🔍 Analyzing image...'):
            # Preprocess image
            img_resized = img.resize((128, 128))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            confidence = float(np.max(prediction))

            # Get disease info
            raw_class = class_names[predicted_index] if predicted_index < len(class_names) else None
            info = disease_info.get(raw_class, {
                "plant": "Unknown",
                "taxonomy": "Unknown",
                "status": "Unknown",
                "disease": "Unknown",
                "cause": "Unknown",
                "deficiency": "Unknown",
                "diagnosis": "Unknown"
            })

        # Show results
        st.markdown("---")
        st.markdown(f"### 🪴 Plant: **{info['plant']}**  _(Taxonomy: *{info['taxonomy']}*)_")
        st.markdown(f"### 🌿 Status: **{info['status']}**")
        st.markdown(f"### 🦠 Disease: **{info['disease']}**")
        st.markdown(f"### 📌 Cause: {info['cause']}")
        st.markdown(f"### 🥕 Nutrient Deficiency: {info['deficiency']}")
        st.markdown(f"### 🧪 Diagnosis: {info['diagnosis']}")

        st.markdown(f"### 🧠 Confidence: **{confidence * 100:.2f}%**")
        if confidence < 0.7:
            st.warning("⚠️ Low confidence — please consider manual verification.")

else:
    st.info("👆 Please upload a leaf image to start diagnosis.")

# Footer
st.markdown("---")
st.markdown(
    "<center>Developed by jaydish kennedy.j<br>AI-ML Developer</center>",
    unsafe_allow_html=True
)



