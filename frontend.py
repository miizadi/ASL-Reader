# frontend.py

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import subprocess
from backend import Backend

class EmotionTrackerApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("800x680")
        self.window.configure(bg="#f0f0f0")

        # Video capture source
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Canvas for video frames
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack(pady=20)

        # Quit button
        self.btn_quit = ttk.Button(window, text="Quit", command=self.quit)
        self.btn_quit.pack(pady=10)

        # Label to display detected emotion
        self.emotion_label = ttk.Label(
            window, text="Detected Emotion: ", font=("Helvetica", 14)
        )
        self.emotion_label.pack(pady=10)

        # Initialize the backend
        self.backend = Backend()

        self.delay = 15  # Delay between frame updates (milliseconds)
        self.update()

        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)

            # Process the frame using the backend
            emotions = self.backend.process_frame(frame)

            # Draw rectangles and labels on the frame
            for (x, y, w, h, emotion_label) in emotions:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    emotion_label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                # Update the emotion label in the GUI
                self.emotion_label.config(text=f"Detected Emotion: {emotion_label}")

                if emotion_label == "Angry":
                    self.send_notification()

            # Convert the frame to RGB and display it in the GUI
            self.photo = ImageTk.PhotoImage(
                image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            )
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Schedule the next frame update
        self.window.after(self.delay, self.update)

    def send_notification(self):
        """
        Retrieves a calming message from the backend and displays a notification.
        """
        calming_message = self.backend.get_calming_message()
        # For macOS
        subprocess.run([
            "osascript",
            "-e",
            f'display notification "{calming_message}" with title "Emotion Alert"'
        ])
        # For Windows (uncomment if using Windows and comment out the macOS command)
        # from win10toast import ToastNotifier
        # toaster = ToastNotifier()
        # toaster.show_toast("Emotion Alert", calming_message, duration=5)

    def quit(self):
        """
        Quits the application.
        """
        self.window.quit()
        self.vid.release()
        cv2.destroyAllWindows()

# Run the application
if __name__ == "__main__":
    EmotionTrackerApp(tk.Tk(), "Emotion Tracker")
