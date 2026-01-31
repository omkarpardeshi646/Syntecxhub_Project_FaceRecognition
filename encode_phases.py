import face_recognition
import os
import pickle

known_encodings = []
known_names = []

dataset_path = "dataset"

for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person)

data = {"encodings": known_encodings, "names": known_names}

with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print("✅ Face encodings saved")
