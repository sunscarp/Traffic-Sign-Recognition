import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign App")

        self.model = load_model('traffic_recognition.h5')
        with open('class_indices.json', 'r') as f:
            self.indices = json.load(f)

        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.root, text="Traffic Sign Recognition")
        self.label.pack(pady=10)

        self.img_label = tk.Label(self.root)
        self.img_label.pack()

        self.browse_button = tk.Button(self.root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=10)

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_sign)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack(pady=10)

    def browse_image(self):
        path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if path:
            self.show_image(path)

    def show_image(self, path):
        img = Image.open(path).resize((224, 224))
        self.img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self.img)
        self.image_path = path
        self.result_label.config(text="")

    def predict_sign(self):
        if hasattr(self, 'image_path'):
            prediction = self.predict_from_image(self.image_path)
            prediction = int(prediction)
            classes = {0: 'Speed limit 5km/h', 1: 'Speed limit 15km/h', 2: 'Speed limit 30km/h', 3: 'Speed limit 40km/h', 4: 'Speed limit 50km/h', 5: 'Speed limit 60km/h', 6: 'Speed limit 70km/h', 7: 'Speed limit 80km/h', 8: 'No left', 9: 'No right', 10: 'No straight', 11: 'No left turn', 12: 'No right turn', 13: 'No U-turn', 14: 'No overtake', 15: 'No U-turn', 16: 'No car', 17: 'No horn', 18: 'Speed limit 40km/h', 19: 'Speed limit 50km/h', 20: 'Go straight or right', 21: 'Go straight', 22: 'Go left', 23: 'Go left or right', 24: 'Go right', 25: 'Keep left', 26: 'Keep right', 27: 'Roundabout', 28: 'Watch out', 29: 'Horn', 30: 'Bicycles crossing', 31: 'U-turn', 32: 'Road divider', 33: 'Traffic signals', 34: 'Danger ahead', 35: 'Zebra crossing', 36: 'Bicycles crossing', 37: 'Children crossing', 38: 'Dangerous curve left', 39: 'Dangerous curve right', 40: 'Unknown1', 41: 'Unknown2', 42: 'Unknown3', 43: 'Go right or straight', 44: 'Go left or straight', 45: 'Unknown4', 46: 'Zigzag curve', 47: 'Train crossing', 48: 'Under construction', 49: 'Unknown5', 50: 'Fences', 51: 'Heavy vehicle accidents', 52: 'Unknown6', 53: 'Give way', 54: 'No stopping', 55: 'No entry', 56: 'Unknown7', 57: 'Unknown8'}
            final_pred = classes[prediction]
            self.result_label.config(text=f"Predicted Sign: {final_pred}")
        else:
            self.result_label.config(text="Select an image first.")

    def predict_from_image(self, path):
        try:
            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            predictions = self.model.predict(img_array)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_name = self.indices[str(predicted_index)]
            return predicted_name
        except Exception as e:
            return f"Error predicting: {e}"

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()
