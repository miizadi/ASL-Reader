import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def load_dataset(dataset_path):
    features = []
    labels = []
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    for emotion_dir in os.listdir(dataset_path):
        emotion_label = emotion_dir
        emotion_dir_path = os.path.join(dataset_path, emotion_dir)
        if not os.path.isdir(emotion_dir_path):
            continue
        for img_name in os.listdir(emotion_dir_path):
            img_path = os.path.join(emotion_dir_path, img_name)
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                continue
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                # We can extract the landmarks
                landmarks = results.multi_face_landmarks[0]  # Assuming one face per image
                # Convert landmarks to a flat list
                feature_vector = []
                for lm in landmarks.landmark:
                    feature_vector.extend([lm.x, lm.y, lm.z])
                features.append(feature_vector)
                labels.append(emotion_label)
            else:
                print(f"No face detected in image {img_path}")
    face_mesh.close()
    return np.array(features), np.array(labels)

def main():
    # Hardcoded dataset path
    dataset_path = '/Users/main/Downloads/archive/train'  # Replace this with your actual dataset path
    
    # Load dataset
    print("Loading dataset...")
    X, y = load_dataset(dataset_path)
    print("Dataset loaded.")
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    # Train classifier
    print("Training classifier...")
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    print("Classifier trained.")
    # Evaluate on test data
    test_accuracy = clf.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy}")
    # Start webcam and predict
    print("Starting webcam...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            # For each face
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh annotations on the image.
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                # Extract landmarks
                feature_vector = []
                for lm in face_landmarks.landmark:
                    feature_vector.extend([lm.x, lm.y, lm.z])
                feature_vector = np.array(feature_vector).reshape(1, -1)
                # Predict emotion
                prediction = clf.predict(feature_vector)
                proba = clf.predict_proba(feature_vector)
                emotion = label_encoder.inverse_transform(prediction)[0]
                confidence = np.max(proba)
                # Display emotion on the frame
                cv2.putText(
                    frame,
                    f"{emotion} ({confidence*100:.1f}%)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        # Display the frame
        cv2.imshow("Emotion Recognition", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
