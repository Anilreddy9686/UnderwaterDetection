from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
import mysql.connector
import os
import traceback
import uuid
import base64
from werkzeug.utils import secure_filename
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ultralytics"))
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import time
from threading import Lock
from functools import wraps
import json
import secrets
from datetime import timedelta
import hmac
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('FLASK_HTTPS_ONLY', '').lower() == 'true',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)

def bounded_setting_from_value(value, default, minimum, maximum):
    try:
        return min(max(float(value), minimum), maximum)
    except (TypeError, ValueError):
        return default

INFERENCE_CONFIDENCE = bounded_setting_from_value(os.environ.get('DETECTION_CONFIDENCE', 0.25), 0.25, 0.05, 0.95)
INFERENCE_IOU = bounded_setting_from_value(os.environ.get('DETECTION_IOU', 0.45), 0.45, 0.10, 0.95)

# ======================== DATABASE SETUP ========================
mydb = None
mycursor = None
db_lock = Lock()
try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        port="3306",
        database="under_water"
    )
    mycursor = mydb.cursor()
    print("✅ Database connected")
    try:
        schema_cursor = mydb.cursor()
        schema_cursor.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                role VARCHAR(32) NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        # Upgrade the original project table in place; existing accounts remain usable.
        schema_cursor.execute("ALTER TABLE users MODIFY name VARCHAR(100) NOT NULL")
        schema_cursor.execute("ALTER TABLE users MODIFY email VARCHAR(255) NOT NULL")
        schema_cursor.execute("ALTER TABLE users MODIFY password VARCHAR(255) NOT NULL")
        try:
            schema_cursor.execute("ALTER TABLE users ADD COLUMN role VARCHAR(32) NOT NULL DEFAULT 'user'")
        except mysql.connector.Error as migration_error:
            if getattr(migration_error, 'errno', None) != 1060:
                raise
        schema_cursor.execute(
            """CREATE TABLE IF NOT EXISTS detections (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                user_id INT NULL,
                source_type VARCHAR(24) NOT NULL,
                source_file VARCHAR(255) NOT NULL,
                model_file VARCHAR(255) NOT NULL,
                detection_count INT NOT NULL DEFAULT 0,
                detections_json LONGTEXT NOT NULL,
                output_file VARCHAR(255) NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_detections_user_date (user_id, created_at),
                CONSTRAINT fk_detections_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )"""
        )
        schema_cursor.execute(
            """CREATE TABLE IF NOT EXISTS audit_logs (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                user_id INT NULL,
                action VARCHAR(100) NOT NULL,
                details VARCHAR(1000) NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_audit_user_date (user_id, created_at)
            )"""
        )
        schema_cursor.execute(
            """CREATE TABLE IF NOT EXISTS app_settings (
                setting_key VARCHAR(64) PRIMARY KEY,
                setting_value VARCHAR(64) NOT NULL,
                updated_by INT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )"""
        )
        schema_cursor.execute("SELECT setting_key, setting_value FROM app_settings")
        for setting_key, setting_value in schema_cursor.fetchall():
            if setting_key == 'confidence':
                INFERENCE_CONFIDENCE = bounded_setting_from_value(setting_value, 0.25, 0.05, 0.95)
            elif setting_key == 'iou':
                INFERENCE_IOU = bounded_setting_from_value(setting_value, 0.45, 0.10, 0.95)
        configured_admin = os.environ.get('ADMIN_EMAIL', '').strip().lower()
        if configured_admin:
            schema_cursor.execute(
                "UPDATE users SET role='admin' WHERE LOWER(email)=%s",
                (configured_admin,)
            )
        mydb.commit()
        schema_cursor.close()
        print("✅ User, history, and audit schema ready")
    except Exception as schema_error:
        print(f"⚠️ Database feature setup incomplete: {schema_error}")
except mysql.connector.Error as e:
    print(f"⚠️ DB connection failed: {e}. App will run without DB features.")

def executionquery(query, values):
    if mycursor is None:
        return
    try:
        with db_lock:
            mycursor.execute(query, values)
            mydb.commit()
    except Exception as e:
        print(f"DB error: {e}")
        traceback.print_exc()

def retrivequery1(query, values):
    if mycursor is None:
        return []
    try:
        with db_lock:
            mycursor.execute(query, values)
            return mycursor.fetchall()
    except Exception as e:
        print(f"DB error: {e}")
        return []

def audit_event(action, details=''):
    if mydb is None:
        return
    try:
        with db_lock:
            cursor = mydb.cursor()
            cursor.execute(
                "INSERT INTO audit_logs (user_id, action, details) VALUES (%s, %s, %s)",
                (session.get('user_id'), action[:100], str(details)[:1000])
            )
            mydb.commit()
            cursor.close()
    except Exception as e:
        print(f"Audit log error: {e}")

def save_detection_record(source_type, source_file, detections, output_file=None, detection_count=None):
    if mydb is None:
        return
    try:
        with db_lock:
            cursor = mydb.cursor()
            cursor.execute(
                """INSERT INTO detections
                   (user_id, source_type, source_file, model_file, detection_count, detections_json, output_file)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (session.get('user_id'), source_type[:24], os.path.basename(source_file)[:255],
                 os.path.basename(MODEL_PATH)[:255], len(detections) if detection_count is None else detection_count,
                 json.dumps(detections, ensure_ascii=False), output_file)
            )
            mydb.commit()
            cursor.close()
    except Exception as e:
        print(f"Detection history save error: {e}")

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please login first.", "warning")
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped

def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please login first.", "warning")
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            return "Administrator access required.", 403
        return view(*args, **kwargs)
    return wrapped

@app.context_processor
def inject_csrf_token():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_urlsafe(32)
    return {'csrf_token': lambda: session['csrf_token']}

@app.before_request
def protect_state_changes():
    if request.method != 'POST':
        return None
    expected = session.get('csrf_token', '')
    supplied = request.form.get('csrf_token', '') or request.headers.get('X-CSRF-Token', '')
    if not expected or not supplied or not hmac.compare_digest(expected, supplied):
        return jsonify({'success': False, 'error': 'Security token missing or expired. Refresh the page and try again.'}), 400
    return None

# ======================== YOLO MODEL LOADING ========================
BASE_MODEL_PATH = os.path.join(BASE_DIR, "best_model+CSEA.pt")
EXPANDED_MODEL_PATH = os.path.join(BASE_DIR, "best_model_expanded.pt")
MODEL_PATH = EXPANDED_MODEL_PATH if os.path.isfile(EXPANDED_MODEL_PATH) else BASE_MODEL_PATH
model = None
model_lock = Lock()

try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Loaded YOLO model from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Failed to load model from {MODEL_PATH}: {e}")
    if MODEL_PATH != BASE_MODEL_PATH and os.path.isfile(BASE_MODEL_PATH):
        try:
            model = YOLO(BASE_MODEL_PATH)
            print(f"⚠️ Using original YOLO model from {BASE_MODEL_PATH}")
        except Exception as fallback_error:
            print(f"❌ Could not load the original model: {fallback_error}")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Cache for recent predictions (to avoid reprocessing similar frames)
prediction_cache = {}
CACHE_SIZE = 10

# ======================== ROUTES ========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if mydb is None:
            flash("Registration requires the database. Start MySQL and try again.", "danger")
            return redirect(url_for('register'))
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        c_password = request.form.get('c_password', '')

        if not name or len(name) > 100 or not email or len(email) > 255:
            flash("Enter a valid name and email address.", "danger")
            return redirect(url_for('register'))
        if len(password) < 8:
            flash("Use a password with at least 8 characters.", "danger")
            return redirect(url_for('register'))

        if password != c_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('register'))

        existing = retrivequery1("SELECT id FROM users WHERE LOWER(email)=%s", (email,))
        if existing:
            flash("Email already registered!", "warning")
            return redirect(url_for('register'))

        executionquery(
            "INSERT INTO users (name, email, password, role) VALUES (%s, %s, %s, 'user')",
            (name, email, generate_password_hash(password))
        )
        audit_event('user.registered', email)
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = retrivequery1(
            "SELECT id, name, email, password, role FROM users WHERE LOWER(email)=%s LIMIT 1",
            (email,)
        )
        valid_user = False
        if user:
            stored_password = user[0][3] or ''
            try:
                valid_user = check_password_hash(stored_password, password)
            except (ValueError, TypeError):
                valid_user = False
            # Existing plaintext accounts are upgraded transparently on successful login.
            if not valid_user and stored_password == password:
                valid_user = True
                executionquery(
                    "UPDATE users SET password=%s WHERE id=%s",
                    (generate_password_hash(password), user[0][0])
                )
        if user and valid_user:
            session.clear()
            session.permanent = True
            session['user_id'] = user[0][0]
            session['user_name'] = user[0][1]
            session['user_email'] = user[0][2]
            session['role'] = user[0][4] or 'user'
            audit_event('user.login', email)
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            audit_event('user.login_failed', email)
            flash("Invalid email or password!", "danger")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout', methods=['POST'])
def logout():
    if 'user_id' in session:
        audit_event('user.logout')
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user_name=session['user_name'])

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        if mydb is None:
            flash("Profile changes require the database.", "danger")
            return redirect(url_for('profile'))
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        if not name or len(name) > 100 or not email or len(email) > 255:
            flash("Enter a valid name and email address.", "danger")
            return redirect(url_for('profile'))
        duplicate = retrivequery1(
            "SELECT id FROM users WHERE LOWER(email)=%s AND id<>%s LIMIT 1",
            (email, session['user_id'])
        )
        if duplicate:
            flash("That email address is already in use.", "danger")
            return redirect(url_for('profile'))
        executionquery("UPDATE users SET name=%s, email=%s WHERE id=%s", (name, email, session['user_id']))
        session['user_name'] = name
        session['user_email'] = email
        audit_event('user.profile_updated', email)
        flash("Profile updated.", "success")
        return redirect(url_for('profile'))
    return render_template('profile.html', user_name=session.get('user_name'), user_email=session.get('user_email'))

@app.route('/history')
@login_required
def history():
    if mydb is None:
        records = []
        flash("History is unavailable because the database is not connected.", "warning")
    elif session.get('role') == 'admin':
        records = retrivequery1(
            "SELECT id, user_id, source_type, source_file, model_file, detection_count, detections_json, output_file, created_at FROM detections ORDER BY created_at DESC LIMIT 250",
            ()
        )
    else:
        records = retrivequery1(
            "SELECT id, user_id, source_type, source_file, model_file, detection_count, detections_json, output_file, created_at FROM detections WHERE user_id=%s ORDER BY created_at DESC LIMIT 100",
            (session['user_id'],)
        )
    return render_template('history.html', records=records)

@app.route('/history/export.csv')
@login_required
def export_history():
    if mydb is None:
        return "Database is not connected.", 503
    import csv
    from io import StringIO
    if session.get('role') == 'admin':
        records = retrivequery1(
            "SELECT id, user_id, source_type, source_file, model_file, detection_count, created_at FROM detections ORDER BY created_at DESC LIMIT 5000",
            ()
        )
    else:
        records = retrivequery1(
            "SELECT id, user_id, source_type, source_file, model_file, detection_count, created_at FROM detections WHERE user_id=%s ORDER BY created_at DESC LIMIT 1000",
            (session['user_id'],)
        )
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['id', 'user_id', 'source_type', 'source_file', 'model_file', 'detection_count', 'created_at'])
    writer.writerows(records)
    audit_event('history.exported', f'{len(records)} records')
    return app.response_class(
        output.getvalue(), mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=detection_history.csv'}
    )

@app.route('/reports')
@login_required
def reports():
    if mydb is None:
        entries = []
    elif session.get('role') == 'admin':
        entries = retrivequery1(
            "SELECT source_type, source_file, detection_count, detections_json, created_at, model_file FROM detections ORDER BY created_at DESC LIMIT 100",
            ()
        )
    else:
        entries = retrivequery1(
            "SELECT source_type, source_file, detection_count, detections_json, created_at, model_file FROM detections WHERE user_id=%s ORDER BY created_at DESC LIMIT 100",
            (session['user_id'],)
        )
    report_rows = []
    for source_type, source_file, detection_count, details_json, created_at, model_file in entries:
        try:
            details = json.loads(details_json or '[]')
        except (ValueError, TypeError):
            details = []
        per_class = {}
        for item in details:
            class_name = item.get('class_name', 'unknown')
            per_class[class_name] = per_class.get(class_name, 0) + int(item.get('count', 1))
        report_rows.append({
            'source_type': source_type, 'source_file': source_file,
            'detection_count': detection_count, 'per_class': per_class,
            'created_at': created_at, 'model_file': model_file,
        })
    return render_template('reports.html', report_rows=report_rows)

@app.route('/admin')
@admin_required
def admin_dashboard():
    user_count = (retrivequery1("SELECT COUNT(*) FROM users", ()) or [(0,)])[0][0]
    detection_count = (retrivequery1("SELECT COUNT(*) FROM detections", ()) or [(0,)])[0][0] if mydb else 0
    latest_users = retrivequery1("SELECT id, name, email, role, created_at FROM users ORDER BY id DESC LIMIT 25", ())
    latest_detections = retrivequery1(
        "SELECT id, user_id, source_type, source_file, detection_count, created_at FROM detections ORDER BY created_at DESC LIMIT 25",
        ()
    ) if mydb else []
    return render_template(
        'admin.html', user_count=user_count, detection_count=detection_count,
        latest_users=latest_users, latest_detections=latest_detections,
        inference_confidence=INFERENCE_CONFIDENCE, inference_iou=INFERENCE_IOU
    )

@app.route('/admin/settings', methods=['POST'])
@admin_required
def update_inference_settings():
    global INFERENCE_CONFIDENCE, INFERENCE_IOU
    if mydb is None:
        flash("Settings require the database connection.", "danger")
        return redirect(url_for('admin_dashboard'))
    try:
        confidence = float(request.form.get('confidence', ''))
        iou = float(request.form.get('iou', ''))
        if not (0.05 <= confidence <= 0.95 and 0.10 <= iou <= 0.95):
            raise ValueError
    except (TypeError, ValueError):
        flash("Confidence must be 0.05–0.95 and IoU must be 0.10–0.95.", "danger")
        return redirect(url_for('admin_dashboard'))
    with db_lock:
        cursor = mydb.cursor()
        cursor.execute(
            "INSERT INTO app_settings (setting_key, setting_value, updated_by) VALUES ('confidence', %s, %s) "
            "ON DUPLICATE KEY UPDATE setting_value=VALUES(setting_value), updated_by=VALUES(updated_by)",
            (str(confidence), session['user_id'])
        )
        cursor.execute(
            "INSERT INTO app_settings (setting_key, setting_value, updated_by) VALUES ('iou', %s, %s) "
            "ON DUPLICATE KEY UPDATE setting_value=VALUES(setting_value), updated_by=VALUES(updated_by)",
            (str(iou), session['user_id'])
        )
        mydb.commit()
        cursor.close()
    INFERENCE_CONFIDENCE, INFERENCE_IOU = confidence, iou
    audit_event('admin.inference_settings_changed', f'confidence={confidence}; iou={iou}')
    flash("Detection thresholds saved.", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/users/<int:user_id>/role', methods=['POST'])
@admin_required
def update_user_role(user_id):
    role = request.form.get('role', '')
    if role not in {'user', 'admin'}:
        return "Invalid role.", 400
    if user_id == session['user_id'] and role != 'admin':
        return "You cannot remove your own administrator role.", 400
    executionquery("UPDATE users SET role=%s WHERE id=%s", (role, user_id))
    audit_event('admin.role_changed', f'user_id={user_id}; role={role}')
    flash("User role updated.", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/prediction')
@login_required
def prediction_page():
    names = getattr(model, 'names', {}) if model is not None else {}
    supported_classes = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    return render_template(
        'prediction.html',
        model_file=os.path.basename(MODEL_PATH) if model is not None else 'Model unavailable',
        supported_classes=supported_classes,
    )

@app.route('/video-analysis')
@login_required
def video_analysis_page():
    return render_template('video_analysis.html')

@app.route('/api/capabilities')
@login_required
def capabilities():
    names = getattr(model, 'names', {}) if model is not None else {}
    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names)]
    else:
        class_names = list(names)
    return jsonify({
        'model_loaded': model is not None,
        'model_file': os.path.basename(MODEL_PATH) if model is not None else None,
        'supported_classes': class_names,
        'confidence_threshold': INFERENCE_CONFIDENCE,
        'iou_threshold': INFERENCE_IOU,
        'inputs': ['image', 'browser_camera', 'video'],
        'estimated_measurements': ['visual visibility and color only'],
        'unavailable_without_calibration_or_labels': [
            'rock detection', 'chemical water quality', 'depth', 'physical object size'
        ],
    })

@app.route('/api/visual-assessment', methods=['POST'])
@login_required
def visual_assessment():
    uploaded = request.files.get('file')
    if uploaded is None or not uploaded.filename:
        return jsonify({'success': False, 'error': 'Choose an image first.'}), 400
    raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'success': False, 'error': 'The selected file is not a readable image.'}), 400
    small = cv2.resize(image, (min(512, image.shape[1]), min(512, image.shape[0])))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    b_mean, g_mean, r_mean = [float(v) for v in cv2.mean(small)[:3]]
    if contrast < 22:
        contrast_label = 'low contrast; visibility may be visually limited'
    elif contrast < 48:
        contrast_label = 'moderate image contrast'
    else:
        contrast_label = 'high image contrast'
    if brightness < 60:
        lighting_label = 'dark image'
    elif brightness > 205:
        lighting_label = 'bright image'
    else:
        lighting_label = 'moderate image brightness'
    audit_event('image.visual_assessed', secure_filename(uploaded.filename))
    return jsonify({
        'success': True,
        'assessment_type': 'visual image-quality heuristic; not a physical water measurement',
        'brightness_mean_0_255': round(brightness, 1),
        'contrast_std_0_255': round(contrast, 1),
        'lighting': lighting_label,
        'contrast_assessment': contrast_label,
        'color_channel_means_bgr': {'blue': round(b_mean, 1), 'green': round(g_mean, 1), 'red': round(r_mean, 1)},
        'limitations': 'This image cannot establish chemical water quality, turbidity in NTU, depth, or actual underwater visibility distance.',
    })

# ======================== YOLO INFERENCE ENDPOINT (Optimized) ========================
@app.route('/prediction', methods=['POST'])
@login_required
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded. Please check server.'}), 500
        
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400
    extension = os.path.splitext(secure_filename(file.filename))[1].lower()
    if extension not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
        return jsonify({'success': False, 'error': 'Upload a JPG, PNG, WEBP, or BMP image.'}), 400
    
    # Generate unique ID for this request
    img_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_FOLDER, f"{img_id}.jpg")
    file.save(temp_path)
    
    try:
        # Read image
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({'success': False, 'error': 'Invalid image file'}), 400
        
        # Run inference with YOLO (optimized for speed)
        with model_lock:  # Ensure thread safety
            results = model(temp_path, conf=INFERENCE_CONFIDENCE, iou=INFERENCE_IOU, verbose=False)
        
        result = results[0]
        
        # Draw bounding boxes on image
        annotated_img = img.copy()
        detections = []
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Get class name
                class_name = model.names[cls_id] if hasattr(model, 'names') else f"Class_{cls_id}"
                
                # Draw rectangle
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (232, 141, 79), 2)
                
                # Draw label background
                label = f"{class_name} {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (232, 141, 79), -1)
                
                # Draw label text
                cv2.putText(annotated_img, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                detections.append({
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })
        
        # Save annotated image
        out_path = os.path.join(OUTPUT_FOLDER, f"{img_id}_annotated.jpg")
        cv2.imwrite(out_path, annotated_img)
        save_detection_record(
            'image', file.filename, detections, f"{img_id}_annotated.jpg"
        )
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        # Update cache
        if len(prediction_cache) >= CACHE_SIZE:
            # Remove oldest entry
            oldest_key = next(iter(prediction_cache))
            del prediction_cache[oldest_key]
        
        return jsonify({
            'success': True,
            'annotated_image': f'/outputs/{img_id}_annotated.jpg',
            'detections': detections,
            'original_filename': file.filename,
            'total_detections': len(detections)
        })
        
    except Exception as e:
        print(f"Prediction error: {traceback.format_exc()}")
        # Clean up on error
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/prediction/video', methods=['POST'])
@login_required
def predict_video():
    if model is None:
        return jsonify({'success': False, 'error': 'Detection model is not loaded.'}), 503
    uploaded = request.files.get('file')
    if uploaded is None or not uploaded.filename:
        return jsonify({'success': False, 'error': 'Choose a video file first.'}), 400
    extension = os.path.splitext(secure_filename(uploaded.filename))[1].lower()
    if extension not in {'.mp4', '.avi', '.mov', '.mkv'}:
        return jsonify({'success': False, 'error': 'Use an MP4, AVI, MOV, or MKV video.'}), 400

    job_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f'{job_id}{extension}')
    output_name = f'{job_id}_annotated.mp4'
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    uploaded.save(input_path)
    capture = cv2.VideoCapture(input_path)
    writer = None
    counts = {}
    sampled_detections = []
    track_state = []
    track_number = 0
    frame_index = 0
    sample_stride = 1
    max_frames = 1800
    try:
        if not capture.isOpened():
            return jsonify({'success': False, 'error': 'The uploaded video could not be opened.'}), 400
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and 1 <= fps <= 120 else 20.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            return jsonify({'success': False, 'error': 'The video has invalid dimensions.'}), 400
        sample_stride = max(1, int(round(fps)))  # detect once per second; repeat annotated frame between samples
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
        )
        if not writer.isOpened():
            return jsonify({'success': False, 'error': 'Could not create the annotated MP4 output.'}), 500

        last_annotated = None
        while frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_stride == 0 or last_annotated is None:
                annotated = frame.copy()
                with model_lock:
                    result = model(frame, conf=INFERENCE_CONFIDENCE, iou=INFERENCE_IOU, verbose=False)[0]
                current_tracks = []
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        confidence = float(box.conf.item())
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        class_name = model.names.get(cls_id, f'Class_{cls_id}') if isinstance(model.names, dict) else model.names[cls_id]
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        candidate = next((
                            old for old in track_state
                            if old['class_name'] == class_name
                            and abs(old['center'][0] - center[0]) + abs(old['center'][1] - center[1]) < 100
                        ), None)
                        if candidate:
                            track_id = candidate['track_id']
                        else:
                            track_number += 1
                            track_id = f'T{track_number:04d}'
                        current_tracks.append({'class_name': class_name, 'center': center, 'track_id': track_id})
                        counts[class_name] = counts.get(class_name, 0) + 1
                        label = f'{class_name} {confidence:.2f} {track_id}'
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (232, 141, 79), 2)
                        cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                        sampled_detections.append({
                            'class_name': class_name, 'confidence': confidence,
                            'bbox': [x1, y1, x2, y2], 'track_id': track_id,
                            'frame': frame_index,
                        })
                track_state = current_tracks
                last_annotated = annotated
            writer.write(last_annotated)
            frame_index += 1
        if frame_index == 0:
            return jsonify({'success': False, 'error': 'The video contains no readable frames.'}), 400
        summary = [{'class_name': name, 'count': count} for name, count in sorted(counts.items())]
        save_detection_record('video', uploaded.filename, summary, output_name, sum(counts.values()))
        audit_event('video.processed', f'{uploaded.filename}; frames={frame_index}')
        return jsonify({
            'success': True,
            'video_url': url_for('serve_video_output', filename=output_name),
            'original_filename': secure_filename(uploaded.filename),
            'processed_frames': frame_index,
            'sample_interval_seconds': 1,
            'summary': summary,
            'detections': sampled_detections[:500],
            'tracking_note': 'Track IDs are approximate within this video and can change after long occlusions.',
        })
    except Exception as e:
        print(f"Video inference error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        try:
            os.remove(input_path)
        except OSError:
            pass

@app.route('/outputs/video/<filename>')
@login_required
def serve_video_output(filename):
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(OUTPUT_FOLDER, safe_filename)
    if safe_filename != filename or not os.path.isfile(filepath):
        return "File not found", 404
    return send_file(filepath, mimetype='video/mp4', conditional=True)

@app.route('/outputs/<filename>')
@login_required
def serve_output(filename):
    """Serve annotated images"""
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(OUTPUT_FOLDER, safe_filename)
    if safe_filename == filename and os.path.isfile(filepath):
        return send_file(filepath, mimetype='image/jpeg', conditional=True)
    else:
        return "File not found", 404

# ======================== HEALTH CHECK ========================
@app.route('/health')
def health():
    """Check if server is running and model is loaded"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH if model else None
    })

if __name__ == '__main__':
    # Run with debug=False for better performance
    app.run(debug=True)

