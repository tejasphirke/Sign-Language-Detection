import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Load the pre-trained model
model = load_model('inceptionv3_model.h5')

# Define the labels for the classes
labels = ['drink','computer', 'before', 'go']

def predict_class(frame):
    # Preprocess the frame
    img = cv2.resize(frame, (240, 240))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make the prediction
    predictions = model.predict(img)
    class_index = np.argmax(predictions[0])
    class_label = labels[class_index]

    return class_label

def main():
    st.title("Class Detection")
    input_mode = st.selectbox("Select input mode", ["Upload video", "Real-time camera"])

    if input_mode == "Upload video":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            cap = cv2.VideoCapture(uploaded_file.name)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                class_label = predict_class(frame)
                st.write(f"Predicted class: {class_label}")
                st.image(frame, channels="BGR", use_column_width=True)

                if st.button('Stop'):
                    break

            cap.release()

    elif input_mode == "Real-time camera":
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            class_label = predict_class(frame)
            st.write(f"Predicted class: {class_label}")
            st.image(frame, channels="BGR", use_column_width=True)

            if st.button('Stop'):
                break

        cap.release()

if __name__ == "__main__":
    main()

