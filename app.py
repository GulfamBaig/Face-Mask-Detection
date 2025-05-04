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
    .download-progress {
        margin: 1rem 0;
        padding: 1rem;
        background: #f5f5f5;
        border-radius: 5px;
    }
    .progress-bar {
        height: 20px;
        background: #e0e0e0;
        border-radius: 10px;
        margin-top: 10px;
    }
    .progress-fill {
        height: 100%;
        background: #4CAF50;
        border-radius: 10px;
        width: 0%;
        transition: width 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# Google Drive model download with enhanced progress tracking
@st.cache_resource(ttl=24*60*60)  # Cache for 24 hours
def load_model():
    # Google Drive file ID (replace with your actual file ID)
    file_id = "1yREfq0xN6pglc9Bo3yMj8qm8RZM4dY-Y"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Local path to save the model
    model_path = "mask_detection_model.h5"
    
    # Create a container for download progress
    progress_container = st.empty()
    
    # Download with progress tracking if not exists
    if not os.path.exists(model_path):
        try:
            # Initialize progress display
            progress_container.markdown("""
            <div class="download-progress">
                <h4>Downloading Model (Large File - Please Wait)</h4>
                <div>Preparing download...</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress"></div>
                </div>
                <div id="progress-text">0% (0 MB/0 MB)</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress callback function
            def progress_callback(current, total, width=80):
                percent = current / total * 100
                mb_current = current / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                progress_container.markdown(f"""
                <div class="download-progress">
                    <h4>Downloading Model (Large File - Please Wait)</h4>
                    <div>Download in progress...</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {percent:.1f}%"></div>
                    </div>
                    <div>{percent:.1f}% ({mb_current:.1f} MB/{mb_total:.1f} MB)</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Download with progress
            gdown.download(url, model_path, quiet=False, callback=progress_callback)
            
            # Clear progress container after download
            progress_container.empty()
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            progress_container.error(f"❌ Failed to download model: {str(e)}")
            return None
    
    try:
        # Show loading message for large model
        with st.spinner('Loading model (this may take a minute for large models)...'):
            model = tf.keras.models.load_model(model_path)
            model.make_predict_function()  # Optimize for inference
            return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

def main():
    st.title("😷 Face Mask Detection")
    st.markdown("""
    Upload an image to detect whether people are wearing face masks.
    This app uses a MobileNetV2 model loaded from Google Drive.
    """)
    
    # Load model (only once per session)
    if 'model' not in st.session_state:
        st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.error("Model could not be loaded. Please try again later.")
        return
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Detect Masks'):
            with st.spinner('Analyzing image...'):
                try:
                    # Simple face detection (center crop)
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
                    (mask_prob, no_mask_prob) = st.session_state.model.predict(face_array, verbose=0)[0]
                    
                    # Draw results
                    draw = ImageDraw.Draw(image)
                    label = "Mask" if mask_prob > no_mask_prob else "No Mask"
                    color = "green" if label == "Mask" else "red"
                    prob = max(mask_prob, no_mask_prob) * 100
                    
                    draw.rectangle([left, top, left + face_size, top + face_size], 
                                 outline=color, width=3)
                    draw.text((left, top - 25), 
                             f"{label}: {prob:.1f}%", 
                             fill=color)
                    
                    # Show results
                    st.subheader("Detection Result")
                    st.image(image, use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.info("Please try another image")

if __name__ == "__main__":
    # Verify requirements
    try:
        import gdown
    except ImportError:
        st.error("Missing required packages. Please add 'gdown' to your requirements.txt")
        st.stop()
    
    main()
