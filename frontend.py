# frontend.py

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import subprocess
import time
from backend import Backend

class EmotionTrackerApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("800x750")
        self.window.configure(bg="#f0f0f0")

        # Video capture source
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Canvas for video frames
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack(pady=20)

        # Frame for control buttons
        self.control_frame = ttk.Frame(window)
        self.control_frame.pack(pady=10)

        # Start button
        self.btn_start = ttk.Button(
            self.control_frame, text="Start", command=self.start_tracking
        )
        self.btn_start.grid(row=0, column=0, padx=5)

        # Stop button
        self.btn_stop = ttk.Button(
            self.control_frame, text="Stop", command=self.stop_tracking
        )
        self.btn_stop.grid(row=0, column=1, padx=5)

        # Quit button
        self.btn_quit = ttk.Button(
            self.control_frame, text="Quit", command=self.quit
        )
        self.btn_quit.grid(row=0, column=2, padx=5)

        # Label to display detected emotion
        self.emotion_label = ttk.Label(
            window, text="Detected Emotion: ", font=("Helvetica", 14)
        )
        self.emotion_label.pack(pady=10)

        # Initialize the backend
        self.backend = Backend()

        self.delay = 15  # Delay between frame updates (milliseconds)

        # Initialize last notification time to 0
        self.last_notification_time = 0

        # Tracking state
        self.is_tracking = False

        # Start the update loop
        self.update()

        self.window.mainloop()

    def start_tracking(self):
        """
        Starts the facial tracking.
        """
        self.is_tracking = True
        self.emotion_label.config(text="Detected Emotion: Tracking started.")

    def stop_tracking(self):
        """
        Stops the facial tracking.
        """
        self.is_tracking = False
        self.emotion_label.config(text="Detected Emotion: Tracking stopped.")

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)

            if self.is_tracking:
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
                    self.emotion_label.config(
                        text=f"Detected Emotion: {emotion_label}"
                    )

                    if emotion_label == "Angry":
                        self.send_notification()
            else:
                # If not tracking, just display the frame without processing
                self.emotion_label.config(text="Detected Emotion: Not tracking.")

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
        Only sends a notification if 10 seconds have passed since the last one.
        """
        current_time = time.time()
        if current_time - self.last_notification_time >= 10:
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

            # Update the last notification time
            self.last_notification_time = current_time

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
