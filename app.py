from flask import Flask, render_template, request, Response,flash,redirect,url_for
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
#from keras.models import load_model
import threading
from flask_login import LoginManager, UserMixin
import os
from flask_login import LoginManager, current_user, login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import model_from_json
app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost/wpdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True)
    email = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))

class Upload(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vtype = db.Column(db.String(255), unique=True)
    video = db.Column(db.String(255), unique=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
      
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists.')
            return redirect(url_for('signup'))
        
        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password, method='sha256'),
           
        )
        db.session.add(new_user)
        db.session.commit()
        
        return render_template("login.html")
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if not user or not check_password_hash(user.password, password):
            return render_template("login.html", aa="Invalid Email or Password")
        
        login_user(user)
        return redirect(url_for('menu'))
    
    return render_template('login.html')


@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        vtype = request.form['vtype']
        return render_template("predictpage.html", vtype=vtype)


@app.route('/menu', methods=['GET', 'POST'])
def menu():
    return render_template('menu.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))



#model = load_model('model.h5')

# Define the activities (classes)
#activities = ['Ak47', 'Gun', 'Knife', 'Sickle', 'Sword'] # Replace with your actual activity labels

# Global variables for video streaming
video_frame = None
emotion=None
video_stream = cv2.VideoCapture()

process_thread = None  # Global variable for the process thread
stop_processing = False  # Flag variable to indicate when to stop processing

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")


def process_video():
    global video_frame, video_stream, stop_processing

    while not stop_processing:
        ret, frame = video_stream.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)            
            emotion =emotion_dict[maxindex]
            #print(emotion_dict[maxindex])
        # Update the global video frame for streaming
        video_frame = frame.copy()

@app.route('/predictpage')
def predictpage():
    # Stop the process thread if it's running
    global stop_processing, process_thread

    if process_thread and process_thread.is_alive():
        stop_processing = True
        process_thread.join()
        stop_processing = False

    return render_template('predictpage.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global video_frame

    while True:
        if video_frame is not None:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', video_frame)
            frame = buffer.tobytes()

            # Yield the frame in the byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/predict', methods=['POST','GET'])
def predict():
    global video_stream, process_thread, stop_processing, video_frame,emotion

    # Get the uploaded video file
    vtype = request.form['vtype']

    video_file = request.files['video']

    # Save the uploaded video file
    video_path = 'static/uploads/' + video_file.filename
    video_file.save(video_path)

    # Release the previous video stream if any
    video_stream.release()

    # Reset the video_frame to None
    video_frame = None
    
    # Load the video
    video_stream = cv2.VideoCapture(video_path)

    # Stop the process thread if it's running
    if process_thread and process_thread.is_alive():
        stop_processing = True
        process_thread.join()
        stop_processing = False

    # Start processing the video in a separate thread
    process_thread = threading.Thread(target=process_video)
    process_thread.start()

    return render_template('result.html', video_path='/video_feed',vtype=vtype,emo=emotion)


if __name__ == '__main__':
    app.run(debug=True)
