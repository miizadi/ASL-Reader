import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import subprocess
import requests
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["API_KEY"])

# Create the model
generation_config = {
  "temperature": 2,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 50,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "give a bunch of one liners that will calm the reader down in the ultimate attempt of them to not get angry and angrier",
      ],
    },
    {
      "role": "model",
      "parts": [
        "## One-Liners to Calm the Reader's Rage:\n\n**General:**\n\n* Breathe. It's just a moment. \n* This too shall pass.\n* Let it go. It's not worth the anger.\n*  Is this really worth your energy? \n*  Focus on what you can control.\n*  There's always a better way to respond.\n\n**For specific situations:**\n\n* **Traffic Jam:** Patience, my friend.  Everyone is in the same boat.\n* **Rude Person:** Their behavior reflects more on them than you. \n* **Missed Opportunity:**  There are always more chances. \n* **Disappointment:** Don't let it define your day. \n* **Stressful Work:** You're doing great. Take a break.\n* **Arguments:**  Listen to understand, not to reply.\n\n**Humorous:**\n\n*  Is anger getting the best of you?  Maybe you should try a nap.\n*  You're not the only one who's ever been angry.  You're in good company!\n*  Take a deep breath and count to ten... in binary. \n*  Remember, a little anger can be good for the immune system. \n*  Rage is just another way to say \"I'm feeling overwhelmed.\"\n\n**Philosophical:**\n\n*  Anger is like a fire.  Don't let it consume you.\n*  The only thing that conquers anger is love. \n*  Let go of the need to be right. \n*  Peace starts within. \n\n**Remember, calming down takes time.  These one-liners are meant to be a starting point.  Take a moment, breathe, and choose your response wisely.** \n",
      ],
    },
  ]
)

def getresponse():
  response = chat_session.send_message("Give a calming one-liner. No quotes or punctuation.")
  return response.text

# Reference to the existing face detection and emotion recognition code
# main.py
startLine: 1
endLine: 60

class EmotionTrackerApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("800x680")
        self.window.configure(bg="#f0f0f0")

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.vid.set(3, 640)
        self.vid.set(4, 480)

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack(pady=20)

        self.btn_quit = ttk.Button(window, text="Quit", command=self.quit)
        self.btn_quit.pack(pady=10)

        self.emotion_label = ttk.Label(window, text="Detected Emotion: ", font=("Helvetica", 14))
        self.emotion_label.pack(pady=10)

        self.model = load_model('face_model.h5')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        self.delay = 15
        self.update()

        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.process_frame(frame)

            # Calculate the dimensions of the canvas and the frame
            canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
            frame_width, frame_height = frame.shape[1], frame.shape[0]

            # Calculate the x and y coordinates to center the image
            x = (canvas_width - frame_width) // 2
            y = (canvas_height - frame_height) // 2

            # Create the PhotoImage object
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            # Display the image on the canvas at the calculated position
            self.canvas.create_image(x, y, image=self.photo, anchor=tk.NW)

            self.window.after(self.delay, self.update)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)  # 1 for horizontal flip

            self.process_frame(frame)

            # Calculate the dimensions of the canvas and the frame
            canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
            frame_width, frame_height = frame.shape[1], frame.shape[0]

            # Calculate the x and y coordinates to center the image
            x = (canvas_width - frame_width) // 2
            y = (canvas_height - frame_height) // 2

            # Create the PhotoImage object
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            # Display the image on the canvas at the calculated position
            self.canvas.create_image(x, y, image=self.photo, anchor=tk.NW)

            self.window.after(self.delay, self.update)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            face_image = cv2.resize(face_roi, (48, 48))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image = np.expand_dims(face_image, axis=0)
            face_image = np.expand_dims(face_image, axis=-1)

            predictions = self.model.predict(face_image)
            emotion_label = self.class_names[np.argmax(predictions)]
            self.emotion_label.config(text=f"Detected Emotion: {emotion_label}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if emotion_label == "Angry":
                self.send_notification()

    def send_notification(self):
        calming_message = self.get_calming_message()
        subprocess.run(["osascript", "-e", f'display notification "{calming_message}" with title "Emotion Alert"'])

    def get_calming_message(self):
        return getresponse()

    def quit(self):
        self.window.quit()

# Create a window and pass it to the Application object
EmotionTrackerApp(tk.Tk(), "Emotion Tracker")