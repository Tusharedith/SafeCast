import base64
import datetime
import json
import math
import os
import threading
import time
from collections import defaultdict
import cv2
import tempfile
import os
import logging
import numpy as np
from flask import Flask, Response, redirect, render_template, request, session
from flask_mail import Mail, Message
from ultralytics import YOLO
from collections import deque
from services.firebase import auth, db, storage
from config import config
from services.chat import get_chat_response
from Crowd_Anamoly.panic_crowd_monitoring import PanicCrowdMonitoring
from google.cloud import exceptions as gcp_exceptions
import logging
import firebase_admin
from firebase_admin import credentials, storage as admin_storage

if not firebase_admin._apps:
    cred = credentials.Certificate("SafeCast/serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {"storageBucket": "suraksha-ai-78e95.firebasestorage.app"})

classNames = []
thread_objects = []
json_path = os.path.join(
    os.path.dirname(__file__), "data", "detection_classes.json"
)
with open(json_path, "r") as f:
    data = json.load(f)
    classNames = data["class_names"]
    thread_objects = data["threat_objects"]

app = Flask(__name__)
app.secret_key = "secret"
app.app_context().push()

app.config["MAIL_SERVER"] = config.MAIL_SERVER
app.config["MAIL_PORT"] = config.MAIL_PORT
app.config["MAIL_USERNAME"] = config.MAIL_USERNAME
app.config["MAIL_PASSWORD"] = config.MAIL_PASSWORD
app.config["MAIL_USE_TLS"] = config.MAIL_USE_TLS
app.config["MAIL_USE_SSL"] = config.MAIL_USE_SSL

model = YOLO("yolo11n.pt")
mail = Mail(app)

conversation_histories = {}


# Store last 3 seconds at 30 FPS (adjust FPS if needed)
FRAME_BUFFER = deque(maxlen=90)  # 3 seconds * 30 fps


@app.route("/chat", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    user_id = session.get("user", {}).get("localId")

    if not user_id:
        return "Please log in to use the chat feature."

    # Reset conversation history for this user on each request
    conversation_history = []

    response, updated_history = get_chat_response(msg, conversation_history, user_id)

    # Store the updated conversation history in the global variable
    conversation_histories[user_id] = updated_history

    return response


def send_email(msg, subject, sender, recipients):
    try:
        msg = Message(subject, sender=sender, recipients=recipients, body=msg)
        mail.send(msg)
    except Exception as e:
        print(e)


def send_email_in_thread(msg, subject, sender, recipients):
    def run_in_context():
        with app.app_context():
            send_email(msg, subject, sender, recipients)

    thread = threading.Thread(target=run_in_context)
    thread.start()


def get_cameras() -> list:
    cameras = []
    index = 0
    while True:
        try:
            camera = cv2.VideoCapture(index)
            if not camera.isOpened():
                break
            cameras.append(index)
            camera.release()
            index += 1
        except Exception as e:
            print(f"Error accessing camera {index}: {str(e)}")
            break
    return cameras


def send_analytics(data: dict, userId: str) -> None:
    if len(data) == 0:
        return

    if userId is None or userId == "":
        return

    data_in_millis = round(time.time() * 1000)

    try:
        db.child("analytics").child(userId).child(data_in_millis).set(data)
    except Exception as e:
        print(e)


def upload_frame_to_firebase(frame, user_id, timestamp, folder="records"):
    # Convert the frame to PNG image data
    _, buffer = cv2.imencode(".png", frame)
    image_data = buffer.tobytes()

    # Create a unique filename for the frame
    filename = f"{user_id}/{folder}/{timestamp.replace(' ', '_').replace(':', '-')}.png"

    try:
        # Upload the frame to Firebase Storage
        bucket = admin_storage.bucket()
        blob = bucket.blob(filename)
        blob.upload_from_string(image_data, content_type="image/png")
        logging.info(f"Successfully uploaded image to {filename}")
    except Exception as e:
        logging.error(f"Failed to upload image to Firebase: {str(e)}")
        raise


def get_images(user_id):
    images = []
    try:
        storage.child(user_id).get_url(user_id)
    except Exception as e:
        print(e)

    return images


@app.route("/capture", methods=["POST"])
def capture():
    data_url = request.json["img_data"]
    img_data = base64.b64decode(data_url.split(",")[1])
    nparry = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparry, cv2.IMREAD_COLOR)
    period = datetime.datetime.now()

    upload_frame_to_firebase(
        img,
        session["user"]["localId"],
        period.strftime("%Y-%m-%d %H:%M:%S"),
        folder="captures",
    )

    return "success"


def gen_frames(user_id, user_email):
    try:
        # Try to open the camera
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera. Please check if it's connected and accessible.")
            return
        
        # Optimize camera settings
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        next_analytics_time = datetime.datetime.now()
        next_detection_time = datetime.datetime.now()
        analytics_delta = datetime.timedelta(seconds=5)
        detection_delta = datetime.timedelta(milliseconds=10)
        objectData = {}
        email_sent = False
        last_detection_results = None

        while True:
            ret, frame = camera.read()
            if not ret:
                break
            FRAME_BUFFER.append(frame.copy())
            current_time = datetime.datetime.now()
            success, frame = camera.read()

            if not success:
                print("Error: Could not read frame from camera")
                break

            frame = cv2.flip(frame, 1)
            output_frame = frame.copy()

            # Only run detection periodically
            if current_time >= next_detection_time:
                try:
                    results = model(frame, stream=True, verbose=False)
                    last_detection_results = results
                    next_detection_time = current_time + detection_delta
                except Exception as e:
                    print(f"Error in object detection: {str(e)}")
                    last_detection_results = None

            objectsFreq = defaultdict(list)

            if last_detection_results:
                for r in last_detection_results:
                    boxes = r.boxes
                    for box in boxes:
                        # bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        confidence = math.ceil((box.conf[0] * 100)) / 100

                        if confidence < 0.5:
                            continue

                        cls = int(box.cls[0])
                        cls_name = classNames[cls]

                        # Draw rectangle and text
                        color = (0, 0, 255) if cls_name in thread_objects else (255, 0, 0)
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            output_frame,
                            f"{cls_name} {confidence * 100:.1f}%",
                            (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                        objectsFreq[cls_name].append(confidence)

                        # Handle threat object detection and email
                        if cls_name in thread_objects and not email_sent:
                            email_thread = threading.Thread(
                                target=handle_threat_detection,
                                args=(
                                    frame,
                                    user_id,
                                    user_email,
                                    cls_name,
                                    confidence,
                                    current_time,
                                ),
                            )
                            email_thread.start()
                            email_sent = True

            # Handle analytics
            if current_time >= next_analytics_time:
                update_analytics(objectsFreq, objectData, current_time)
                send_analytics(objectData, user_id)
                objectData = {}
                email_sent = False
                next_analytics_time = current_time + analytics_delta

            # Convert frame to bytes and yield
            ret, buffer = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    except Exception as e:
        print(f"Error in video feed: {str(e)}")
    finally:
        if 'camera' in locals():
            camera.release()



def handle_threat_detection(frame, user_id, user_email, cls_name, confidence, timestamp):
    with app.app_context():
        image_filename = f"{user_id}/records/{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.png"
        video_filename = f"{user_id}/records/{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
        

        # Simulate a short video with 3 seconds of the same frame (or use real buffer)
        video_frames = list(FRAME_BUFFER)


        # Upload Image
        try:
            upload_frame_to_firebase(frame, user_id, timestamp.strftime("%Y-%m-%d %H:%M:%S"))

            bucket = admin_storage.bucket()
            blob = bucket.blob(image_filename)

            if not blob.exists():
                raise gcp_exceptions.NotFound(f"Image file not found: {image_filename}")

            image_url = blob.generate_signed_url(datetime.timedelta(hours=1))
        except Exception as e:
            logging.error(f"Image error: {e}")
            image_url = "Image URL not available due to error."

        # Upload Video
        try:
            video_url = upload_video_to_firebase(video_frames, user_id, timestamp)
        except Exception as e:
            logging.error(f"Video upload failed: {e}")
            video_url = "Video URL not available due to error."

        # Send alert email
        send_email(
            f"🚨 Security Alert: Unauthorized Object Detected 🚨\n\n"
            f"🔍 Object: {cls_name.capitalize()}\n"
            f"🕒 Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"🎯 Confidence: {confidence * 100:.1f}%\n\n"
            f"🖼 Image Preview: {image_url}\n"
            f"🎥 Video Clip: {video_url}\n\n"
            f"This is an automated alert.\n\n"
            f"Regards,\nSafeCast Team",
            "Security Alert: Unauthorized Object Detected",
            user_email,
            [user_email],
        )

        logging.info(f"Image saved at: {image_filename}")
        logging.info(f"Video saved at: {video_filename}")

# def handle_threat_detection(
#     frame, user_id, user_email, cls_name, confidence, timestamp
# ):
#     with app.app_context():
#         # Generate a unique filename for the image
#         filename = f"{user_id}/records/{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.png"

#         try:
#             # Upload the frame to Firebase Storage
#             upload_frame_to_firebase(
#                 frame, user_id, timestamp.strftime("%Y-%m-%d %H:%M:%S")
#             )

#             # Verify that the file exists before trying to generate a URL
#             bucket = admin_storage.bucket()
#             blob = bucket.blob(filename)

#             if not blob.exists():
#                 raise gcp_exceptions.NotFound(f"File not found: {filename}")

#             # Generate a signed URL
#             image_url = blob.generate_signed_url(datetime.timedelta(hours=1))

#             if not image_url:
#                 raise Exception("Failed to generate signed URL")

#         except gcp_exceptions.NotFound as e:
#             logging.error(f"File not found in storage: {str(e)}")
#             image_url = "Image not available. File upload may have failed."
#         except Exception as e:
#             logging.error(f"Error uploading image or generating URL: {str(e)}")
#             image_url = "Image URL not available due to technical issues."

#         send_email(
#             f"Security Alert: Unauthorized Object Detected\n\n"
#             f"Object: {cls_name.capitalize()}\n"
#             f"Detection Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
#             f"Confidence Level: {confidence * 100:.1f}%\n\n"
#             f"Description:\n"
#             f"The object '{cls_name}' was detected by our security system. "
#             f"An attempt was made to upload an image for visual confirmation.\n\n"
#             f"Image Status: {image_url}\n\n"
#             f"This is an automated alert.\n\n"
#             f"Regards,\nSafeCast Team",
#             "Security Alert: Unauthorized Object Detected",
#             user_email,
#             [user_email],
#         )

#         # Log the full path that was attempted
#         logging.info(f"Attempted to access file at path: {filename}")


def update_analytics(objectsFreq, objectData, current_time):
    for obj, confidences in objectsFreq.items():
        if confidences:
            if obj in objectData:
                objectData[obj]["freq"] = max(objectData[obj]["freq"], len(confidences))
                objectData[obj]["maxConfidence"] = max(
                    objectData[obj]["maxConfidence"], max(confidences)
                )
                objectData[obj]["minConfidence"] = min(
                    objectData[obj]["minConfidence"], min(confidences)
                )
            else:
                objectData[obj] = {
                    "freq": len(confidences),
                    "maxConfidence": max(confidences),
                    "minConfidence": min(confidences),
                    "time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                
def upload_video_to_firebase(frames, user_id, timestamp):
    try:
        # Save frames as a temporary video
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, f"temp_{timestamp.strftime('%Y%m%d%H%M%S')}.mp4")

        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

        for frame in frames:
            out.write(frame)
        out.release()

        # Upload to Firebase Storage
        filename = f"{user_id}/records/{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
        bucket = admin_storage.bucket()
        blob = bucket.blob(filename)
        blob.upload_from_filename(video_path, content_type='video/mp4')

        # Generate signed URL
        video_url = blob.generate_signed_url(datetime.timedelta(hours=1))
        return video_url

    except Exception as e:
        logging.error(f"Video upload failed: {e}")
        return "Video URL not available due to error."

@app.route("/")
def index():
    return render_template(
        "index.html",
    )


@app.route("/video")
def video():
    if session.get("user") is None:
        return redirect("/signin")

    camera_feed = request.args.get("camera_feed", False) == "True"

    return render_template("video.html", cameras=get_cameras(), camera_feed=camera_feed)


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(
            session["user"]["localId"] if session.get("user") is not None else None,
            session["user"]["email"] if session.get("user") is not None else None,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user") is not None:
        return redirect("/profile")

    if request.method == "POST":
        first_name = request.form["first_name"]
        last_name = request.form["last_name"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if (
            first_name == ""
            or last_name == ""
            or email == ""
            or password == ""
            or confirm_password == ""
        ):
            return render_template("signup.html", error="All fields are required")

        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match")

        user = auth.create_user_with_email_and_password(email, password)
        user = auth.update_profile(
            id_token=user["idToken"], display_name=first_name + " " + last_name
        )

        session["user"] = user

        return redirect("/profile")

    return render_template("signup.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if email == "" or password == "":
            return render_template("signin.html", error="All fields are required")

        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session["user"] = user
            return redirect("/profile")

        except Exception as e:
            return render_template(
                "signin.html", error="Invalid email or password", error_message=str(e)
            )

    return render_template("signin.html")


@app.route("/signout")
def signout():
    session.pop("user", None)
    return render_template("signin.html")


@app.route("/profile")
def profile():
    if session.get("user") is None:
        return redirect("/signin")

    return render_template("profile.html", user=session.get("user"))


@app.route("/dashboard")
def dashboard():
    if session.get("user") is None:
        return redirect("/signin")

    doc = db.child("analytics").child(session["user"]["localId"]).get()
    data = doc.val()

    if data is None:
        data = {}

    # reduce data to 10 most recent entries
    data = dict(list(data.items())[-15:])

    return render_template(
        "dashboard.html",
        user=session.get("user"),
        data=data,
        images=get_images(session["user"]["localId"]),
    )


@app.route("/clear_logs", methods=["POST"])
def clear_logs():
    if session.get("user") is None:
        return redirect("/signin")

    db.child("analytics").child(session["user"]["localId"]).remove()

    return redirect("dashboard")



@app.route('/panic-detection')
def panic_detection():
    return render_template('panic detection.html')

# Define the panic detection frame generator function
def gen_panic_frames():
    """
    Generator function for panic detection video feed.
    This is separate from the regular gen_frames function.
    """
    try:
        # Initialize the panic detection monitor
        panic_monitor = PanicCrowdMonitoring()
        
        # Set up camera
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera for panic detection")
            return
            
        while True:
            # Read frame from camera
            success, frame = camera.read()
            if not success:
                print("Error: Failed to read frame from camera")
                break
                
            # Process the frame with panic detection
            if hasattr(panic_monitor, 'process_frame'):
                result_frame, is_panic = panic_monitor.process_frame(frame)
                
                # Add visual indicator if panic is detected
                if is_panic:
                    cv2.putText(result_frame, "PANIC DETECTED!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                frame = result_frame
            else:
                # Fallback if process_frame doesn't exist
                cv2.putText(frame, "Panic Detection Mode", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Convert to JPEG and yield
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    except Exception as e:
        print(f"Error in panic detection feed: {str(e)}")
    finally:
        if 'camera' in locals():
            camera.release()

# Update the route to use the correct function
@app.route('/panic-video-feed')
def panic_video_feed():
    """Route handler for the panic detection video feed."""
    return Response(gen_panic_frames(),
                  mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=3001)
