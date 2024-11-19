from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import threading
import time
import gym
import atexit

class ZyDash:
    def __init__(
            self, 
            env, 
            agent=None, 
            host='0.0.0.0', 
            port=5000, 
            resize_factor=3,
            old_env=False):

        self.env = env
        self.agent = agent
        self.host = host
        self.port = port
        self.resize_factor = resize_factor
        self.old_env = old_env
        
        self.app = Flask(__name__)
        
        # Control flags
        self.paused = threading.Event()
        self.restart_requested = threading.Event()
        self.step_requested = threading.Event()
        self.paused.set()  # Start paused by default

        # Routes
        self.app.add_url_rule('/', view_func=self.index)
        self.app.add_url_rule('/video_feed', view_func=self.video_feed)
        self.app.add_url_rule('/control', view_func=self.control, methods=['POST'])

        atexit.register(lambda: self.env.close())

        self.env.reset()

    def generate_frames(self):
        while not self.restart_requested.is_set():
            if not self.paused.is_set() or self.step_requested.is_set():
                if self.agent is not None:
                    action = self.agent.get_action(self.env)
                else:
                    action = self.env.action_space.sample()
                
                if self.old_env:
                    obs, reward, done, info = self.env.step(action)
                else:
                    obs, reward, done, truncated, info = self.env.step(action)
                
                # Render the environment
                frame = self.env.render()
                height, width, _ = frame.shape
                new_window_size = (width*self.resize_factor, height*self.resize_factor)
                frame = cv2.resize(frame, new_window_size, interpolation=cv2.INTER_LINEAR)
                _, img = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                # Convert the image to bytes
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img.tobytes() + b'\r\n')
                
                if done:
                    self.env.reset()
                
                time.sleep(0.05)  # Control the frame rate
                self.step_requested.clear()

    def index(self):
        return render_template('index.html')

    def video_feed(self):
        return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def control(self):
        action = request.json.get('action')
        if action == 'start':
            self.paused.clear()
        elif action == 'pause':
            self.paused.set()
        elif action == 'restart':
            self.restart_requested.set()
            self.env.reset()
            self.restart_requested.clear()
        elif action == 'step':
            self.step_requested.set()
        return jsonify(success=True)

    def run(self):
        self.app.run(host=self.host, port=self.port, debug=False)
