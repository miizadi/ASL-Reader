# backend.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import google.generativeai as genai
import os

class Backend:
    def __init__(self):
        # Load the emotion detection model
        self.model = load_model('face_model.h5')
        
        # Load the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Emotion class names
        self.class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Configure the generative AI model
        genai.configure(api_key=os.environ["API_KEY"])
        
        # Create the generation configuration
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 50,
            "response_mime_type": "text/plain",
        }
        
        # Initialize the generative AI model
        self.genai_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )
        
        # Start a chat session with predefined history
        self.chat_session = self.genai_model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        "Give a bunch of one-liners that will calm the reader down to prevent anger escalation.",
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        "Sure, here are some calming one-liners:\n\n- Take a deep breath; this moment will pass.\n- Stay calm; you've handled worse before.\n- Focus on solutions, not problems.\n- Keep your cool; it's not worth the stress.\n- Pause and reset; you control your response.\n- Let go of what's beyond your control.\n- Stay centered; peace begins with you.",
                    ],
                },
            ]
        )
    
    def process_frame(self, frame):
        """
        Detects faces in the frame and predicts the emotion for each face.
        Returns a list of emotions with their corresponding face coordinates.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )
        emotions = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            face_image = cv2.resize(face_roi, (48, 48))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image = np.expand_dims(face_image, axis=0)
            face_image = np.expand_dims(face_image, axis=-1)
            predictions = self.model.predict(face_image)
            emotion_label = self.class_names[np.argmax(predictions)]
            emotions.append((x, y, w, h, emotion_label))
        return emotions

    def get_calming_message(self):
        """
        Retrieves a calming message from the generative AI model.
        """
        response = self.chat_session.send_message(
            "Give a quick tip on how to calm down when using a computer. No special characters. Each response should be different from the last."
        )
        return response.text.strip()
