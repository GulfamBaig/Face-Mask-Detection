import streamlit as st
import numpy as np
import os
import gdown
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 900px;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin-top: 1rem;
    }
    .stAlert {
        border-radius: 5px;
    }
    .header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-container {
        margin-top: 2rem;
        border: 1px solid #eee;
        border-radius: 8px;
        padding: 1rem;
    }
    .progress-container {
        margin: 1rem 0;
        background-color: #f1f1f1;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .progress-text {
        text-align: center;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Google Drive model download with progress tracking
@st.cache_resource(ttl=24*60*60)  # Cache for 24 hours
def load_model():
    # Google Drive file ID (replace with your actual file ID)
    file_id = "1yREfq0xN6pglc9Bo3yMj8qm8RZM4dY-Y"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Local path to save the model
    model_path = "mask_detection_model.h5"
    
    # Download with progress tracking if not exists
    if not os.path.exists(model_path):
        try:
            # Create progress container
            progress_container = st.empty()
            progress_container.markdown("""
            <div class="progress-container">
                <div class="progress-text">Downloading model (this may take several minutes)...</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress callback function
            def progress_callback(current, total, width=80):
                percent = current / total * 100
                progress_container.markdown(f"""
                <div class="progress-container">
                    <div class="progress-text">Downloading model: {percent:.1f}% complete</div>
                    <progress value="{current}" max="{total}" style="width:100%"></progress>
                    <div style="text-align: center; font-size: 0.8rem;">
                        {current/(1024*1024):.1f}MB / {total/(1024*1024):.1f}MB
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Download with progress
            gdown.download(url, model_path, quiet=False, callback=progress_callback)
            
            progress_container.empty()
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {str(e)}")
            return None
    
    try:
        # Show loading message
        with st.spinner('Loading model... This may take a minute for large models'):
            model = tf.keras.models.load_model(model_path)
            model.make_predict_function()  # Optimize for inference
            return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

def detect_faces(image, model):
    """Basic face detection using center crop (replace with better detection if needed)"""
    width, height = image.size
    face_size = min(width, height) // 2
    left = (width - face_size) // 2
    top = (height - face_size) // 2
    
    # Extract face
    face = image.crop((left, top, left + face_size, top + face_size))
    
    # Preprocess
    face = face.resize((224, 224))
    face_array = img_to_array(face)
    face_array = preprocess_input(face_array)
    face_array = np.expand_dims(face_array, axis=0)
    
    # Predict
    (mask_prob, no_mask_prob) = model.predict(face_array, verbose=0)[0]
    
    return {
        'face_box': (left, top, left + face_size, top + face_size),
        'mask_prob': float(mask_prob),
        'no_mask_prob': float(no_mask_prob)
    }

def main():
    st.markdown("<div class='header'><h1>😷 Face Mask Detection</h1></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        Upload an image to detect whether people are wearing face masks.
        The app uses a MobileNetV2 model trained to classify mask/no-mask.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Load model (only once)
    if not st.session_state.model_loaded:
        model = load_model()
        if model is not None:
            st.session_state.model = model
            st.session_state.model_loaded = True
        else:
            st.error("Model could not be loaded. Please try again later.")
            return
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Detect Masks'):
            with st.spinner('Analyzing image...'):
                try:
                    # Detect faces and predict masks
                    result = detect_faces(image, st.session_state.model)
                    
                    # Draw results
                    draw = ImageDraw.Draw(image)
                    box = result['face_box']
                    label = "Mask" if result['mask_prob'] > result['no_mask_prob'] else "No Mask"
                    color = "green" if label == "Mask" else "red"
                    prob = max(result['mask_prob'], result['no_mask_prob']) * 100
                    
                    draw.rectangle(box, outline=color, width=3)
                    draw.text((box[0], box[1] - 25), 
                             f"{label}: {prob:.1f}%", 
                             fill=color, font_size=20)
                    
                    # Show results
                    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                    st.subheader("Detection Result")
                    st.image(image, use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.info("Please try another image or check the console for details")

if __name__ == "__main__":
    # Check requirements
    try:
        import gdown
    except ImportError:
        st.error("Missing required packages. Please add 'gdown' to your requirements.txt")
        st.stop()
    
    main()
