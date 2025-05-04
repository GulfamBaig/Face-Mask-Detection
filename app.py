import streamlit as st
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 for short-range, 1 for full-range
    min_detection_confidence=0.5
)

# Load the pre-trained model
@st.cache_resource
def load_face_mask_model():
    model = load_model('mask_detection_model.h5')
    return model

model = load_face_mask_model()

def detect_mask_with_mediapipe(image, model):
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert BGR to RGB (MediaPipe requires RGB)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    results = face_detection.process(img_rgb)
    
    # Process each detected face
    if results.detections:
        for detection in results.detections:
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img_array.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Ensure coordinates are within image bounds
            x, y = max(0, x), max(0, y)
            w, h = min(iw - x, w), min(ih - y, h)
            
            # Extract face ROI
            face_roi = img_array[y:y+h, x:x+w]
            
            # Skip if face ROI is empty
            if face_roi.size == 0:
                continue
                
            # Preprocess for MobileNetV2
            face_roi = cv2.resize(face_roi, (224, 224))
            face_roi = img_to_array(face_roi)
            face_roi = preprocess_input(face_roi)
            face_roi = np.expand_dims(face_roi, axis=0)
            
            # Make prediction
            (mask, withoutMask) = model.predict(face_roi)[0]
            
            # Determine label and color
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            # Include probability in label
            label = f"{label}: {max(mask, withoutMask) * 100:.1f}%"
            
            # Display rectangle and label
            cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_array, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    return img_array

def main():
    st.set_page_config(
        page_title="Face Mask Detection",
        page_icon="😷",
        layout="centered"
    )
    
    st.title("😷 Face Mask Detection")
    st.markdown("""
    Upload an image to detect whether people are wearing face masks. 
    This app uses a MobileNetV2 model trained to classify mask/no-mask.
    """)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Detect Masks'):
            with st.spinner('Processing image...'):
                try:
                    # Convert to numpy array
                    image_np = np.array(image)
                    
                    # Detect faces and predict masks
                    result_image = detect_mask_with_mediapipe(image_np, model)
                    
                    # Display result
                    st.subheader("Detection Result")
                    st.image(result_image, use_column_width=True)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.warning("Please try another image or check the console for details")

if __name__ == "__main__":
    main()
