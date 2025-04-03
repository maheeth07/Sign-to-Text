from flask import Flask, render_template, Response, jsonify
from keras.models import model_from_json
import cv2
import numpy as np
import mediapipe as mp
from function import *

app = Flask(__name__)

# Load model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")


sequence, sentence, accuracy, predictions = [], [], [], []

mp_hands = mp.solutions.hands
cap = None  # Camera is initially off
camera_running = False

def generate_frames():
    global camera_running
    global sequence, sentence, accuracy, predictions  # Access global variables
    actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])
    threshold = 0.8
    print("cammmmmmm", camera_running)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened() and camera_running:
            ret, frame = cap.read()
            if not ret:
                break
            cropframe = frame[40:400, 0:300]
            frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)

            image, results = mediapipe_detection(cropframe, hands)

            keypoints = extract_keypoints(results)
            print("---keyp---", keypoints)
            if keypoints is not None:  # Add this check
                sequence.append(keypoints)
                sequence = sequence[-30:]  # Keep the last 30 frames

            

            try:
                print("---sequence", sequence, len(sequence) )
                if len(sequence) == 30:
                    print("1 a")
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))
                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        print("2 b")
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                print("3 c")
                                if actions[np.argmax(res)] != sentence[-1]:
                                    print("4 d")
                                    sentence.append(actions[np.argmax(res)])
                                    accuracy.append(str(round(res[np.argmax(res)] * 100, 2)) + '%')
                            else:
                                print("5 e")
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(round(res[np.argmax(res)] * 100, 2)) + '%')

                    if len(sentence) > 1:
                        print("6 f")
                        sentence = sentence[-1:]
                        accuracy = accuracy[-1:]
                        print("---sent", sentence, accuracy)

            except Exception as e:
                print(e)

            cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
            cv2.putText(frame, "Output: " + ' '.join(sentence) + ''.join(accuracy), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            print("-----------------\n", sentence, accuracy, "\n-----------------\n")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_output')
def get_output():
    return jsonify({'sentence': ''.join(sentence), 'accuracy': accuracy[-1] if accuracy else '0%'})

@app.route('/start_camera', methods=['GET'])
def start_camera():
    global cap, camera_running
    if not camera_running:
        cap = cv2.VideoCapture(0)  # Open camera only when starting
        camera_running = True
    return jsonify({"status": "Camera started"})

@app.route('/stop_camera', methods=['GET'])
def stop_camera():
    global cap, camera_running
    if camera_running:
        camera_running = False
        if cap is not None:
            cap.release()  # Release camera when stopping
            cap = None
    return jsonify({"status": "Camera stopped"})

if __name__ == '__main__':
    app.run(debug=True)
