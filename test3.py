import cv2
import numpy as np
import face_recognition
import requests
import os

# Telegram bot details
bot_token = '7377193087:AAF8ILEg0LjO8Ucx9omlYhXRAuR0cDto0FU'
chat_id = '1765902365'

# Function to send photo and message to Telegram
def send_telegram_alert(image_path, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.post(url, data=payload)
    if response.status_code != 200:
        print("Failed to send message:", response.text)

    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    with open(image_path, 'rb') as img:
        files = {'photo': img}
        response = requests.post(url, files=files, data={'chat_id': chat_id})
        if response.status_code != 200:
            print("Failed to send photo:", response.text)

# Load known faces and their encodings
known_face_encodings = []
known_face_names = []

# Load images of known faces
known_faces_dir = 'known_faces'
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if len(encoding) > 0:
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(filename)[0])  # Use the filename (without extension) as the person's name
        else:
            print(f"No faces found in {filename}")

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize camera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)

        # Find face encodings
        face_encodings = face_recognition.face_encodings(rgb_face)
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # Label the face
            cv2.putText(frame, name, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)

            if name == "Unknown":
                # Save the image
                img_name = "unknown_person.jpg"
                cv2.imwrite(img_name, frame)

                # Send alert to Telegram
                print("Unknown person detected! Sending alert...")
                send_telegram_alert(img_name, "A person is detetected in you office permises please check and verify this [Thank you Cloud Education R&D Team ]")
                os.remove(img_name)
        else:
            print("No face encodings found in the ROI.")

    # Display the resulting frame
    cv2.imshow('Security Camera', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
