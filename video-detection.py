import cv2
from ultralytics import YOLO
import os

def detect_objects_in_downloaded_video(video_path):
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        return None
    
    try:
        yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None
    
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        print("Possible reasons: File is corrupted, unsupported codec, or OpenCV lacks codec support.")
        return None
    
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fps = video_capture.get(cv2.CAP_PROP_FPS)  
    print(f"Video properties: Width={width}, Height={height}, FPS={fps}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    result_video_path = "./Results/detected_downloaded_video.mp4"
    out = cv2.VideoWriter(result_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not initialize video writer. Check codec support.")
        video_capture.release()
        return None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        try:
            results = yolo_model(frame)
        except Exception as e:
            print(f"Error during YOLO inference: {e}")
            break

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
    print(f"Processed video saved as MP4 to: {result_video_path}")
    
    return result_video_path

if __name__ == "__main__":
    test_dir = './test_vid'
    video_file = "vid1.mp4"
    video_path = os.path.join(test_dir, video_file)
    
    result = detect_objects_in_downloaded_video(video_path)
    if not result:
        print("Processing failed. Check error messages above for details.")