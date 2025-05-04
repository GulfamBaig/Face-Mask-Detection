import os
import gdown
import tensorflow as tf
import streamlit as st

# Google Drive model download function
@st.cache_resource
def load_model():
    # Google Drive file ID (replace with your actual file ID)
    file_id = "1yREfq0xN6pglc9Bo3yMj8qm8RZM4dY-Y"
    
    # You can also use the direct download link if preferred
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Local path to save the model
    model_path = "mask_detection_model.h5"
    
    # Download if not exists
    if not os.path.exists(model_path):
        try:
            st.info("Downloading model from Google Drive... This may take a while.")
            gdown.download(url, model_path, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            st.error("Please check: ")
            st.error("- Your internet connection")
            st.error("- The file ID is correct and accessible")
            st.error("- You have sufficient storage space")
            return None
    
    try:
        st.info("Loading model...")
        model = tf.keras.models.load_model(model_path)
        
        # For TensorFlow 2.x, make_predict_function isn't usually needed
        # but we'll keep it for compatibility
        if hasattr(model, 'make_predict_function'):
            model.make_predict_function()  # Optimize for inference
            
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.error("Possible issues:")
        st.error("- Corrupted model file (try deleting and re-downloading)")
        st.error("- Incompatible TensorFlow version")
        st.error("- Missing dependencies")
        return None

# Load the model
model = load_model()

# Check if model loaded successfully before proceeding
if model is None:
    st.warning("Could not load model. The app may not function properly.")
else:
    # Your model-dependent code here
    st.write("Model is ready for predictions!")
