from picamera2 import Picamera2
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import time

# Load class labels (using simple text file format)
with open("efficientdet_lite0_metadata.json", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize Pi Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Load TFLite model
interpreter = tflite.Interpreter(model_path="efficientdet_lite0.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

try:
    while True:
        # Capture frame
        frame = picam2.capture_array("main")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Preprocess (UINT8 version)
        img = cv2.resize(frame, (320, 320))
        img = img.astype(np.uint8)  # Keep in 0-255 range
        img = np.expand_dims(img, axis=0)
        
        # Set input and run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        
        # Get results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        
        # Draw detections with labels
        for i in range(len(scores)):
            if scores[i] > 0.5:  # Confidence threshold
                y1, x1, y2, x2 = boxes[i]
                x1 = int(x1 * frame.shape[1])
                y1 = int(y1 * frame.shape[0])
                x2 = int(x2 * frame.shape[1])
                y2 = int(y2 * frame.shape[0])
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Get class name and confidence
                class_id = int(classes[i])
                class_name = labels[class_id]
                confidence = scores[i]
                
                # Display label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
