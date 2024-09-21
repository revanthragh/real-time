# Import necessary libraries
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tkinter as tk
from tkinter import Label, Canvas, Button, Frame, filedialog, OptionMenu, StringVar

# Load the trained emotion recognition model
model_path = 'Model.h5'
model = load_model(model_path)

# Load Haarcascades face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define the emotion classes
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define the main application class
class EmotionRecognitionApp:
    def __init__(self, window, window_title):
        # Initialize the main window and its components
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x800")

        # Create the main frame
        self.frame = Frame(window, bg='#001759')
        self.frame.pack(expand=True, fill='both')

        # Create and pack labels, buttons, and dropdown menu
        self.title_label = Label(self.frame, text="Emotion Recognition App", font=("Helvetica", 24), bg='white', )
        self.title_label.pack(pady=(20, 10))

        self.canvas = Canvas(self.frame, width=1200, height=700, bg='white')
        self.canvas.pack(side=tk.LEFT, padx=10)

        self.label = Label(self.frame, text="", font=("Helvetica", 16), justify="center", anchor="w", bg='#001759')
        self.label.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)

        self.start_button = Button(self.frame, text="Start Camera", command=self.start_emotion_recognition,
                                   font=("Helvetica", 16), bg='#4CAF50', fg='white', padx=20, pady=10)
        self.start_button.pack(side=tk.TOP, padx=10, pady=10)

        self.load_image_button = Button(self.frame, text="Load Image", command=self.load_image,
                                        font=("Helvetica", 16), bg='#4CAF50', fg='white', padx=20, pady=10)
        self.load_image_button.pack(side=tk.TOP, padx=10, pady=10)

        self.load_video_button = Button(self.frame, text="Load Video", command=self.load_video,
                                        font=("Helvetica", 16), bg='#4CAF50', fg='white', padx=20, pady=10)
        self.load_video_button.pack(side=tk.TOP, padx=10, pady=10)

        self.emotion_var = StringVar(self.frame)
        self.emotion_var.set("All Emotions")
        self.emotion_menu = OptionMenu(self.frame, self.emotion_var, "All Emotions", *emotion_classes)
        self.emotion_menu.config(font=("Helvetica", 16), bg='#4CAF50', fg='white')
        self.emotion_menu.pack(side=tk.TOP, padx=10, pady=10)

        self.window.protocol("WM_DELETE_WINDOW", self.on_window_close)

        self.video_capture = None
        self.emotion_recognition_active = False

    def load_image(self):
        # Open a file dialog to select an image file
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            # Process the selected image
            self.process_image(file_path)

    def process_image(self, file_path):
        # Read the image using OpenCV and process the frame
        frame = cv2.imread(file_path)
        self.process_frame(frame)

    def start_emotion_recognition(self):
        # Start the emotion recognition using the computer's camera
        if not self.emotion_recognition_active:
            self.video_capture = cv2.VideoCapture(0)
            self.emotion_recognition_active = True
            self.update_emotion_recognition()

    def stop_emotion(self):
        # Stop the emotion recognition process
        if self.emotion_recognition_active:
            self.emotion_recognition_active = False
            if self.video_capture is not None:
                self.video_capture.release()

    def on_window_close(self):
        # Handle window close event, stop emotion recognition and destroy the window
        self.stop_emotion()
        self.window.destroy()

    def update_emotion_recognition(self):
        # Continuously update the emotion recognition using the camera feed
        if self.emotion_recognition_active:
            ret, frame = self.video_capture.read()
            if ret:
                self.process_frame(frame)
                self.window.after(10, self.update_emotion_recognition)
            else:
                print("Error capturing frame.")
                self.stop_emotion()

    def process_frame(self, frame):
        # Process each frame from the camera feed
        selected_emotion = self.emotion_var.get() if self.emotion_var.get() != "All Emotions" else None
        self._process_frame(frame, filter_by_emotion=selected_emotion)

    def _process_frame(self, frame, filter_by_emotion=False):
        # Core logic to detect faces, predict emotions, and update UI
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        sentiment_counts = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            resized_roi = cv2.resize(roi_gray, (48, 48))
            img_array = image.img_to_array(resized_roi)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            predictions = model.predict(img_array)

            predicted_class = np.argmax(predictions)

            if 0 <= predicted_class < len(emotion_classes):
                predicted_emotion = emotion_classes[predicted_class]
                sentiment_counts[predicted_emotion] += 1

                if not filter_by_emotion or (filter_by_emotion and predicted_emotion == filter_by_emotion):
                    box_color = (0,255,0 )  # Green
                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

                    # Display emotion and accuracy on the frame with red font color
                    accuracy = predictions[0][predicted_class]
                    font_size = int(w / 10)  # Adjust the divisor for a different font size scaling factor
                    text = f"{predicted_emotion} (Accuracy: {accuracy:.2f})"
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size / 15, (0,255,0),
                                2)  # Red font color

        sentiment_text = "\n".join([f"{emotion}: {count}" for emotion, count in sentiment_counts.items()])
        self.label.config(text=sentiment_text)

        self.display_frame(frame)

    def display_frame(self, frame):
        # Display the processed frame on the Tkinter canvas
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        original_height, original_width, _ = frame_rgb.shape
        aspect_ratio = original_width / original_height

        new_width = 800
        new_height = int(new_width / aspect_ratio)

        resized_frame = cv2.resize(frame_rgb, (new_width, new_height))

        img = Image.fromarray(resized_frame)
        img = ImageTk.PhotoImage(image=img)

        self.canvas.img = img
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)

    def load_video(self):
        # Open a file dialog to select a video file
        file_path = filedialog.askopenfilename(title="Select a video", filetypes=[("Video files", "*.mp4")])
        if file_path:
            # Process the selected video
            self.process_video(file_path)

    def process_video(self, file_path):
        # Read the video file and start emotion recognition
        self.video_capture = cv2.VideoCapture(file_path)
        self.emotion_recognition_active = True
        self.update_emotion_recognition_video()

    def update_emotion_recognition_video(self):
        # Continuously update the emotion recognition using video feed
        if self.emotion_recognition_active:
            ret, frame = self.video_capture.read()
            if ret:
                self.process_frame(frame)
                self.window.after(10, self.update_emotion_recognition_video)
            else:
                print("Video processing complete.")
                self.stop_emotion()

if __name__ == "__main__":
    # Create the main Tkinter window and run the application
    root = tk.Tk()
    app = EmotionRecognitionApp(root, "Emotion Recognition App")
    root.mainloop()
