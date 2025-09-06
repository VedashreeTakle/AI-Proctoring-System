from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import time
import uuid
import json
import base64
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np
import io
import threading
import cv2
import time
import numpy as np
import torch
import speech_recognition as sr
import mediapipe as mp
from ultralytics import YOLO
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
app.config['UPLOAD_FOLDER'] = 'screenshots'
app.config['SESSION_TIMEOUT'] = 30 * 60  # 30 minutes session timeout

# Create screenshots directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Exam questions
EXAM_QUESTIONS = {
    'mcq': [
        {
            'id': 1,
            'question': 'What is the capital of France?',
            'options': ['London', 'Berlin', 'Paris', 'Madrid'],
            'correct': 2  # Index of correct answer (Paris)
        },
        {
            'id': 2,
            'question': 'Which planet is known as the Red Planet?',
            'options': ['Earth', 'Mars', 'Jupiter', 'Venus'],
            'correct': 1  # Index of correct answer (Mars)
        }
    ],
    'paragraph': [
        {
            'id': 3,
            'question': 'Explain the concept of artificial intelligence and its potential impact on society.',
        },
        {
            'id': 4,
            'question': 'Discuss the ethical implications of using facial recognition technology for surveillance.',
        }
    ]
}

@app.route('/')
def index():
    """Landing page"""
    # Generate a unique session ID for non-login authentication
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/permissions')
def permissions():
    """Permission checkpoints page"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    return render_template('permissions.html')

@app.route('/verify_permissions', methods=['POST'])
def verify_permissions():
    """Verify that permissions are granted"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Session expired'})
    
    data = request.json
    # Store permission status in session
    session['permissions'] = {
        'camera': data.get('camera', False),
        'microphone': data.get('microphone', False)
    }
    
    if session['permissions']['camera'] and session['permissions']['microphone']:
        return jsonify({'success': True})
    else:
        return jsonify({
            'success': False, 
            'message': 'Camera and microphone permissions are required'
        })

@app.route('/exam')
def exam():
    """Exam interface"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    if 'permissions' not in session or not session['permissions'].get('camera') or not session['permissions'].get('microphone'):
        return redirect(url_for('permissions'))
    
    # Initialize exam session
    if 'exam_started' not in session:
        session['exam_started'] = time.time()
        session['remaining_time'] = app.config['SESSION_TIMEOUT']
        session['cheating_incidents'] = []
        session['answers'] = {
            'mcq': {},
            'paragraph': {}
        }
    
    return render_template('exam.html', 
                          questions=EXAM_QUESTIONS,
                          remaining_time=session['remaining_time'])


import requests

################################
def trigger_proctoring():
    """Send a request to start the proctoring system"""
    try:
        response = requests.post(
            "http://localhost:5007/check_cheating",  # Changed to match the route in the proctoring app
            json={"action": "start_proctoring", "timestamp":"jjj"},
            timeout=5
        )
        
        if response.status_code == 200:
            print("✅ Successfully triggered proctoring system")
            return True
        else:
            print(f"❌ Failed to trigger proctoring: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to proctoring service: {e}")
        return False

@app.route('/receive_proctoring_data', methods=['POST'])
def receive_proctoring_data():
    """Endpoint to receive proctoring data from the proctoring app"""
    try:
        
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data received"
            }), 400
            
        
        
        return jsonify({
            "status": "success",
            "message": "Evidence data received successfully"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing evidence data: {str(e)}"
        }), 500      
        
    
   
################################

@app.route('/save_answer', methods=['POST'])
def save_answer():
    """Save student's answer"""
    if 'user_id' not in session or 'exam_started' not in session:
        return jsonify({'success': False, 'message': 'Session expired'})
    
    data = request.json
    question_id = data.get('question_id')
    answer = data.get('answer')
    question_type = data.get('type')
    
    if question_id and answer is not None and question_type in ['mcq', 'paragraph']:
        session['answers'][question_type][str(question_id)] = answer
        session.modified = True
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'message': 'Invalid data'})

@app.route('/submit_exam', methods=['POST'])
def submit_exam():
    """Submit the exam"""
    if 'user_id' not in session or 'exam_started' not in session:
        return jsonify({'success': False, 'message': 'Session expired'})
    
    # Calculate time taken
    time_taken = time.time() - session['exam_started']
    
    # Save exam data
    exam_data = {
        'user_id': session['user_id'],
        'start_time': datetime.fromtimestamp(session['exam_started']).strftime("%Y-%m-%d %H:%M:%S"),
        'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'time_taken': time_taken,
        'cheating_incidents': session['cheating_incidents'],
        'answers': session['answers']
    }
    
    # In a real application, you'd save this to a database
    # For now, we'll just save to a JSON file
    log_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session['user_id']}_exam_log.json")
    with open(log_file, 'w') as f:
        json.dump(exam_data, f, indent=4)
    
    # Clear session data
    session.pop('exam_started', None)
    session.pop('remaining_time', None)
    session.pop('cheating_incidents', None)
    session.pop('answers', None)
    
    return jsonify({'success': True, 'redirect': url_for('completion')})

@app.route('/completion')
def completion():
    """Exam completion page"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    return render_template('completion.html')

if __name__ == '__main__':
    trigger_proctoring()
    app.run(debug=True)
    
    