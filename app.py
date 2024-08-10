from flask import Flask, render_template, request, redirect, url_for
import cv2
import face_recognition
from deepface import DeepFace
import os

app = Flask(__name__)

# Load criminal faces and their encodings
criminal_faces_dir = "criminal_faces/"
criminal_encodings = []
criminal_names = []

for filename in os.listdir(criminal_faces_dir):
    image = face_recognition.load_image_file(os.path.join(criminal_faces_dir, filename))
    encoding = face_recognition.face_encodings(image)[0]
    criminal_encodings.append(encoding)
    criminal_names.append(os.path.splitext(filename)[0])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', result="No file uploaded.", emotion="N/A")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', result="No file selected.", emotion="N/A")

        # Save uploaded file
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        # Load and process the image
        image = face_recognition.load_image_file(filepath)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) == 0:
            return render_template('index.html', result="No face detected.", emotion="N/A")

        user_encoding = face_encodings[0]

        # Compare with criminal faces
        matches = face_recognition.compare_faces(criminal_encodings, user_encoding)
        face_distances = face_recognition.face_distance(criminal_encodings, user_encoding)
        best_match_index = face_distances.argmin()

        if matches[best_match_index]:
            match_name = criminal_names[best_match_index]
            result = f"Match found! The user is a criminal: {match_name}."
        else:
            result = "No match found. The user is not a criminal."

        # Detect emotions
        emotion_result = DeepFace.analyze(filepath, actions=['emotion'])
        dominant_emotion = emotion_result['dominant_emotion']

        return render_template('index.html', result=result, emotion=dominant_emotion)

    return render_template('index.html', result=None, emotion=None)

if __name__ == '__main__':
    app.run(debug=True)
