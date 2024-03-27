import streamlit as st
import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('/Users/sreevalsan/Desktop/Project/Emotion detect CNN Project/Sree_emotion_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    resized_img = cv2.resize(image, (48, 48))
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    img_array = np.array(gray_img)  # Convert to NumPy array
    img_array_batch = np.expand_dims(img_array, axis=0)
    img_array_batch_with_channels = np.expand_dims(img_array_batch, axis=-1)
    img_array_batch_with_channels = np.repeat(img_array_batch_with_channels, 3, axis=-1)
    return img_array_batch_with_channels

# Function to predict emotion
def predict_emotion(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return np.argmax(prediction)

# Streamlit app
def main():
    st.title('Emotion Detection')

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Read uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Display uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict emotion on button click
        if st.button('Predict'):
            emotion_index = predict_emotion(image)
            emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
            predicted_emotion = emotion_dict[emotion_index]
            st.success(f'Predicted Emotion: {predicted_emotion}')

if __name__ == '__main__':
    main()
