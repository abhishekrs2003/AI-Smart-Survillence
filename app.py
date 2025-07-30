from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
import cv2
import os
import time
import threading
import sqlite3
import smtplib
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import base64
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DETECTION_IMAGES'] = 'static/detection_images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTION_IMAGES'], exist_ok=True)
app.secret_key = 'your_secret_key'  # Change this to a strong secret key

# Email Configuration
EMAIL_LOGIN = '8821f8001@smtp-brevo.com'
EMAIL_SENDER = 'rsabhishek98@gmail.com'
EMAIL_PASSWORD = 'q5Uwch1ZYpHJdN28'  # Use app password if using Gmail
EMAIL_RECEIVER = 'rsabhishek98@gmail.com'
SMTP_SERVER = 'smtp-relay.brevo.com'
SMTP_PORT = 587

# Detection Configuration
ALERT_THRESHOLD = 0.5  # Only send alerts for detections with confidence above this threshold
COOLDOWN_PERIOD = 60  # Seconds between alerts for the same model to prevent spam

# Database setup
DB_PATH = 'detections.db'

prototype_processing_active = True
prototype_processing_lock = threading.Lock()

@app.route('/start_processing', methods=['POST'])
def start_processing():
    global processing_active, prototype_processing_active
    with processing_lock:
        processing_active = True
    with prototype_processing_lock:
        prototype_processing_active = True
    return jsonify({"message": "Processing started"}), 200

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global processing_active, cap, prototype_processing_active
    with processing_lock:
        processing_active = False
        if cap:
            cap.release()
            cap = None
    with prototype_processing_lock:
        prototype_processing_active = False
    return jsonify({"message": "Processing stopped"}), 200
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        model_type TEXT NOT NULL,
        class_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        image_path TEXT NOT NULL,
        alerted INTEGER DEFAULT 0
    )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Load Models
models = {
    "accident": YOLO("yolo11n_accident_100epoch.pt"),  # Replace with actual model path
    "violence": YOLO("yolo11n_violence_50epoch.pt")   # Replace with actual model path
}

# Use a lock for thread safety when switching models
model_lock = threading.Lock()
model_type = "accident"  # Keep track of current model type

# Flag to control video processing
processing_active = False
processing_lock = threading.Lock()

# Dummy user for authentication
USER_CREDENTIALS = {'admin': '123'}
cap = None
FRAME_WIDTH, FRAME_HEIGHT = 640, 640
CONFIDENCE_THRESHOLD = 0.5  # 50% confidence threshold

# Track last alert time for each model type to implement cooldown
last_alert_time = {"accident": 0, "violence": 0}

def save_detection_to_db(model_type, class_name, confidence, image_path):
    """Save detection information to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO detections (timestamp, model_type, class_name, confidence, image_path) VALUES (?, ?, ?, ?, ?)",
        (timestamp, model_type, class_name, confidence, image_path)
    )
    detection_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return detection_id

def send_alert_email(model_type, class_name, confidence, image_path):
    """Send an email alert with the detection image"""
    global last_alert_time
    
    current_time = time.time()
    # Check if we're still in cooldown period for this model type
    if current_time - last_alert_time.get(model_type, 0) < COOLDOWN_PERIOD:
        print(f"Alert for {model_type} skipped due to cooldown period")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = f"ALERT: {model_type.capitalize()} Detection"
        
        # Email body
        body = f"""
        <html>
        <body>
            <h2>Detection Alert</h2>
            <p>A potential {model_type} has been detected by the system.</p>
            <p><strong>Details:</strong></p>
            <ul>
                <li>Type: {model_type.capitalize()}</li>
                <li>Classification: {class_name}</li>
                <li>Confidence: {confidence:.2f}</li>
                <li>Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</li>
            </ul>
            <p>Please see the attached image for visual confirmation.</p>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))
        
        # Attach the image
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            image = MIMEImage(img_data)
            image.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
            msg.attach(image)
        
        # Connect to server and send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_LOGIN, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        # Update the last alert time for this model type
        last_alert_time[model_type] = current_time
        
        print(f"Alert email sent for {model_type} detection")
        return True
    
    except Exception as e:
        print(f"Failed to send email alert: {str(e)}")
        return False

def mark_detection_as_alerted(detection_id):
    """Mark a detection as alerted in the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE detections SET alerted = 1 WHERE id = ?", (detection_id,))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html', user=session.get('user'))

@app.route('/delete_detection/<int:id>', methods=['POST'])
def delete_detection(id):
    """Delete a detection entry from the database and return a response"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get the image path before deleting
    cursor.execute("SELECT image_path FROM detections WHERE id = ?", (id,))
    row = cursor.fetchone()
    if row:
        image_path = row[0]
        # Delete the image file
        if os.path.exists(image_path):
            os.remove(image_path)

    # Delete detection entry from the database
    cursor.execute("DELETE FROM detections WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    
    return redirect(url_for('detections'))


@app.route('/test')
def test():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('test.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        login_type = request.form.get('login_type')  # Get whether it's user or admin login

        # Define credentials with roles
        USER_CREDENTIALS = {
            'admin': {'password': 'admin123', 'role': 'admin'},
            'user': {'password': 'user123', 'role': 'user'}
        }

        # Validate credentials
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username]['password'] == password:
            user_role = USER_CREDENTIALS[username]['role']

            # Ensure login type matches the expected role
            if login_type == 'admin' and user_role != 'admin':
                return render_template('login.html', error='Invalid login. Use the User Login page.')

            if login_type == 'user' and user_role != 'user':
                return render_template('login.html', error='Invalid login. Use the Admin Login page.')

            # Store session details
            session['user'] = username
            session['role'] = user_role

            return redirect(url_for('index'))  # Redirect after successful login

        return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')



@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/switch_model', methods=['POST'])
def switch_model():
    global model_type
    data = request.get_json()

    if data and 'model_type' in data:
        new_model_type = data['model_type']

        if new_model_type in models:
            with model_lock:
                model_type = new_model_type
            return jsonify({"message": f"Switched to {new_model_type} model"}), 200
        else:
            return jsonify({"error": "Invalid model type"}), 400

    return jsonify({"error": "Bad Request"}), 400

@app.route('/get_current_model')
def get_current_model():
    return jsonify({"current_model": model_type})


@app.route('/detections')
def detections():
    if 'user' not in session:
        return redirect(url_for('login'))  # Redirect if not logged in

    if session.get('role') != 'admin':  
        return "Access Denied: You must be an admin to view this page.", 403  # Restrict access to admin only

    page = request.args.get('page', 1, type=int)
    per_page = 12  # Number of detections per page

    # Get filter parameters
    model_type = request.args.get('model_type', '')
    date_filter = request.args.get('date', '')
    min_confidence = request.args.get('min_confidence', 0.5, type=float)

    # Build query with filters
    query = "SELECT * FROM detections WHERE 1=1"
    params = []

    if model_type:
        query += " AND model_type = ?"
        params.append(model_type)

    if date_filter:
        query += " AND DATE(timestamp) = ?"  # Extract only date from timestamp
        params.append(date_filter)

    if min_confidence:
        query += " AND confidence >= ?"
        params.append(min_confidence)

    # Get total count for pagination
    count_query = "SELECT COUNT(*) FROM detections WHERE 1=1"
    count_params = params.copy()

    if model_type:
        count_query += " AND model_type = ?"

    if date_filter:
        count_query += " AND DATE(timestamp) = ?"

    if min_confidence:
        count_query += " AND confidence >= ?"

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # ✅ Fix: Allows accessing fields as attributes
    cursor = conn.cursor()

    cursor.execute(count_query, count_params)
    total_items = cursor.fetchone()[0]

    # Calculate total pages
    total_pages = (total_items + per_page - 1) // per_page
    page = max(1, min(page, total_pages)) if total_pages > 0 else 1

    # Add pagination to query
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([per_page, (page - 1) * per_page])

    cursor.execute(query, params)
    detections = cursor.fetchall()
    conn.close()

    # Preserve filter values for pagination
    filter_args = {}
    if model_type:
        filter_args['model_type'] = model_type
    if date_filter:
        filter_args['date'] = date_filter
    if min_confidence:
        filter_args['min_confidence'] = min_confidence

    return render_template('detections.html', 
                          detections=detections,
                          page=page,
                          total_pages=total_pages,
                          filter_args=filter_args)



def generate_frames():
    global cap, model_type, processing_active
    if cap is None:
        return

    # Define class name mapping for the accident model
    accident_class_name_mapping = {0: "Potential Accident", 1: "Normal"}
    
    # Variables to track significant detections
    last_detection_time = 0
    detection_cooldown = 2  # Seconds between saving detections to avoid duplicates

    while cap is not None and cap.isOpened():
        # Check if processing should continue
        with processing_lock:
            if not processing_active:
                break
        
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Use thread lock to safely access the current model
        with model_lock:
            current_model_type = model_type
            current_model = models[current_model_type]
        
        results = current_model(frame)
        detection_to_save = None  # Track the highest confidence detection in this frame
        
        # Process the frame and find any detections
        original_frame = frame.copy()  # Keep an unmodified copy for saving
        
        for result in results:
            for box in result.boxes:
                confidence = box.conf[0].item()
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])  # Extract class index

                # Check which model is being used and apply appropriate mapping
                if current_model_type == "accident":
                    class_name = accident_class_name_mapping.get(class_id, "Unknown")
                    # Only save accident detections (class 0), not normal traffic (class 1)
                    if class_id == 0 and (detection_to_save is None or confidence > detection_to_save[2]):
                        detection_to_save = (class_name, (x1, y1, x2, y2), confidence)
                else:
                    class_name = result.names[class_id]  # Use default class name for other models
                    if detection_to_save is None or confidence > detection_to_save[2]:
                        if class_name.lower() == "violence":
                            detection_to_save = (class_name, (x1, y1, x2, y2), confidence)

                box_color = (0, 255, 0)
                text_color = (255, 255, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, lineType=cv2.LINE_AA)
                label_text = f"{class_name} {confidence:.2f}"

                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_offset_x, text_offset_y = x1, y1 - 10

                cv2.rectangle(frame, (text_offset_x, text_offset_y - text_height - 5),
                            (text_offset_x + text_width + 10, text_offset_y), box_color, -1)
                cv2.putText(frame, label_text, (text_offset_x + 5, text_offset_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, lineType=cv2.LINE_AA)
        
        # Save significant detection if found and not in cooldown period
        current_time = time.time()
        if detection_to_save and (current_time - last_detection_time > detection_cooldown):
            class_name, (x1, y1, x2, y2), confidence = detection_to_save
            
            # Skip normal traffic class for accident model
            if not (current_model_type == "accident" and class_name == "Normal"):
                # Save detection image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"{current_model_type}_{timestamp}.jpg"
                image_path = f"static/detection_images/{image_filename}"
                
                # Draw detection on frame for saving
                detection_image = original_frame.copy()
                cv2.rectangle(detection_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(detection_image, f"{class_name} ({confidence:.2f})", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imwrite(image_path, detection_image)
                
                # Save to database
                relative_image_path = image_path.replace("\\", "/")  # Convert to forward slashes
                detection_id = save_detection_to_db(current_model_type, class_name, confidence, relative_image_path)

                
                # Send alert email if confidence is high enough
                if confidence >= ALERT_THRESHOLD:
                    if send_alert_email(current_model_type, class_name, confidence, image_path):
                        mark_detection_as_alerted(detection_id)
                
                last_detection_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/upload', methods=['POST'])
def upload():
    global cap, processing_active
    if 'user' not in session:
        return redirect(url_for('login'))

    if 'video' not in request.files:
        return redirect(request.url)

    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Stop any existing processing
    with processing_lock:
        if cap:
            cap.release()
        cap = cv2.VideoCapture(filepath)
        processing_active = True

    return redirect(url_for('test'))

@app.route('/video_feed')
def video_feed():
    if 'user' not in session:
        return redirect(url_for('login'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prototype_test')
def prototype_test():
    # Renders a dedicated page for prototype accident detection.
    return render_template('prototype_test.html')


def generate_prototype_frames():
    """
    This function uses the prototype accident detection logic (based on your accident.py)
    to capture frames from the camera (index 1), run YOLO segmentation for accident detection,
    send an alert email, and log the detection in the database when an accident is found.
    """
    # Load the YOLO segmentation model for accident detection
    global prototype_processing_active
    prototype_model = YOLO("car_accident.pt")  # Ensure this file exists in your project
    vehicles = {"car", "motorcycle", "bus", "truck"}

    cap_prototype = cv2.VideoCapture(1)
    if not cap_prototype.isOpened():
        print("Prototype accident detection: Unable to open camera")
        return

    last_alert_time_prototype = 0  # local cooldown timer

    while cap_prototype.isOpened():
        with prototype_processing_lock:
            if not prototype_processing_active:
                break
        ret, frame = cap_prototype.read()
        if not ret:
            break

        h_orig, w_orig, _ = frame.shape
        small_frame = cv2.resize(frame, (640, 640))
        results = prototype_model.predict(small_frame, conf=0.5, imgsz=640)

        detected_vehicles = []
        mask_overlay = np.zeros_like(frame, dtype=np.uint8)

        # Process each detection result to extract segmentation masks and bounding boxes
        for result in results:
            if result.masks and result.boxes:
                masks = result.masks.xy  # segmentation masks
                boxes = result.boxes.xyxy  # bounding boxes
                classes = result.boxes.cls  # class IDs
                for mask, box, class_id in zip(masks, boxes, classes):
                    class_id = int(class_id)
                    class_name = prototype_model.names[class_id]
                    if class_name in vehicles:
                        scale_x = w_orig / 640
                        scale_y = h_orig / 640
                        scaled_mask = np.array(mask) * (scale_x, scale_y)
                        x1, y1, x2, y2 = box
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                        detected_vehicles.append((class_name, scaled_mask.astype(np.int32), (x1, y1, x2, y2)))
                        cv2.fillPoly(mask_overlay, [scaled_mask.astype(np.int32)], (255, 255, 255))

        # Blend the overlay with the original frame for visualization
        frame = cv2.addWeighted(frame, 0.8, mask_overlay, 0.2, 0)

        # Check for collisions between detected vehicles using bounding boxes and mask overlap
        accident_detected = False
        collision_info = ""
        for i in range(len(detected_vehicles)):
            name1, mask1, box1 = detected_vehicles[i]
            x1a, y1a, x2a, y2a = box1
            for j in range(i + 1, len(detected_vehicles)):
                name2, mask2, box2 = detected_vehicles[j]
                x1b, y1b, x2b, y2b = box2
                if not (x2a < x1b or x2b < x1a or y2a < y1b or y2b < y1a):
                    mask_img1 = np.zeros(frame.shape[:2], dtype=np.uint8)
                    mask_img2 = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask_img1, [mask1], 255)
                    cv2.fillPoly(mask_img2, [mask2], 255)
                    intersection = cv2.bitwise_and(mask_img1, mask_img2)
                    if np.sum(intersection) > 500:
                        accident_detected = True
                        collision_info = f"{name1} hit {name2}"
                        break
            if accident_detected:
                break

        # If an accident is detected and cooldown has passed, send an email and log the detection
        if accident_detected:
            current_time = time.time()
            if current_time - last_alert_time_prototype >= COOLDOWN_PERIOD:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"prototype_accident_{timestamp}.jpg"
                image_path = f"static/detection_images/{image_filename}"
                cv2.putText(frame, "ACCIDENT DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imwrite(image_path, frame)
                relative_image_path = image_path.replace("\\", "/")
                detection_id = save_detection_to_db("prototype_accident", collision_info, 1.0, relative_image_path)
                if send_alert_email("prototype_accident", collision_info, 1.0, image_path):
                    mark_detection_as_alerted(detection_id)
                last_alert_time_prototype = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap_prototype.release()


@app.route('/prototype_video_feed')
def prototype_video_feed():
    # Streams the processed video feed with prototype accident detection
    return Response(generate_prototype_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True,port=6012)
