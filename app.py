import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the pre-trained model
@st.cache_resource
def load_face_mask_model():
    model = load_model('mask_detection_model.h5')
    return model

model = load_face_mask_model()

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_predict_mask(frame, faceNet, maskNet):
    # Grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI, convert it from BGR to RGB channel ordering,
            # resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.shape[0] > 0 and face.shape[1] > 0:  # Ensure face is not empty
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # Add the face and bounding boxes to their respective lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # Only make a predictions if at least one face was detected
    if len(faces) > 0:
        # For faster inference we'll make batch predictions on all faces at once
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

def detect_mask_simple(image, model):
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = img_array[y:y+h, x:x+w]
        
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
    # Custom CSS for styling
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }
        .header {
            color: #2c3e50;
            text-align: center;
            padding: 10px;
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
        .stRadio>div>div {
            display: flex;
            justify-content: center;
        }
        .stRadio>div>label {
            margin: 0 10px;
        }
        .result-box {
            border-radius: 5px;
            padding: 20px;
            margin: 10px 0;
            background-color: white;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        }
        .image-container {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # App header
    st.markdown("<h1 class='header'>Face Mask Detection</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        Upload an image or use your webcam to detect whether people are wearing face masks.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Settings")
    detection_mode = st.sidebar.radio(
        "Detection Mode",
        ("Image Upload", "Webcam")
    )

    # Main content
    if detection_mode == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button('Detect Masks'):
                with st.spinner('Processing...'):
                    # Convert to numpy array
                    image_np = np.array(image)
                    
                    # Detect faces and predict masks
                    result_image = detect_mask_simple(image_np, model)
                    
                    # Display result
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: center;'>Detection Result</h3>", unsafe_allow_html=True)
                    st.image(result_image, use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
    
    else:  # Webcam mode
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            Click the button below to start your webcam for real-time mask detection.
        </div>
        """, unsafe_allow_html=True)
        
        run = st.checkbox('Start Webcam')
        
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces and predict masks
            result_frame = detect_mask_simple(frame, model)
            
            FRAME_WINDOW.image(result_frame)
        else:
            st.write('Webcam is stopped')

if __name__ == "__main__":
    main()
