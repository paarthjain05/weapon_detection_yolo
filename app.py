import cv2
from ultralytics import YOLO
from flask import Flask, request, send_file, jsonify
import os
import uuid

app = Flask(__name__)

model_path = "runs/detect/Normal_Compressed/weights/best.pt"
try:
    yolo_model = YOLO(model_path)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def detect_objects_in_video(video_path, output_path):
    if yolo_model is None:
        return False, "YOLO model not loaded"

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        return False, "Could not open video file"

    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        video_capture.release()
        return False, "Could not initialize video writer"

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        try:
            results = yolo_model(frame)
        except Exception as e:
            video_capture.release()
            out.release()
            return False, f"YOLO inference failed: {e}"

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        out.write(frame)

    video_capture.release()
    out.release()
    return True, "Success"

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(('.mp4', '.avi')):
        return jsonify({"error": "Unsupported file format. Use MP4 or AVI."}), 400

    unique_id = str(uuid.uuid4())
    input_filename = f"{unique_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    file.save(input_path)

    output_filename = f"{unique_id}_detected.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    success, message = detect_objects_in_video(input_path, output_path)

    try:
        os.remove(input_path)
    except Exception as e:
        print(f"Error deleting input file: {e}")

    if not success:
        return jsonify({"error": message}), 500

    try:
        response = send_file(output_path, mimetype='video/mp4', as_attachment=True, download_name=output_filename)
        os.remove(output_path)
        return response
    except Exception as e:
        return jsonify({"error": f"Error sending file: {e}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": yolo_model is not None}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)