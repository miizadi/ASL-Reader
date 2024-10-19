import mediapipe as mp
import cv2

# Initialize MediaPipe face mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Create a face mesh object
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
) as face_mesh:

    # Capture video from your webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()

        # Convert BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = face_mesh.process(image)

        # Draw the detected face mesh
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(thickness=1),
                )

        # Display the image
        cv2.imshow('Face Mesh', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()