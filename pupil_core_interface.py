# pupil_core_interface.py
import zmq
import msgpack
import threading
import queue

class PupilCore:
    def __init__(self, ip='127.0.0.1', port=50020):
        self.ip = ip
        self.port = port
        self.context = zmq.Context()
        self.remote = self.context.socket(zmq.REQ)
        self.remote.connect(f'tcp://{ip}:{port}')
        self.subscriber = None
        self.gaze_queue = queue.Queue()
        self.running = False
        self.listener_thread = None

    def _get_sub_port(self):
        self.remote.send_string('SUB_PORT')
        return self.remote.recv_string()

    def start_gaze_listener(self):
        sub_port = self._get_sub_port()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(f'tcp://{self.ip}:{sub_port}')
        self.subscriber.subscribe('gaze.')
        self.running = True
        self.listener_thread = threading.Thread(target=self._listen, daemon=True)
        self.listener_thread.start()

    def _listen(self):
        while self.running:
            try:
                subscriber = self.subscriber
                if subscriber is None:
                    continue
                topic, payload = subscriber.recv_multipart()
                data = msgpack.unpackb(payload, raw=False)
                norm_pos = data.get(b'norm_pos', None) or data.get('norm_pos', None)
                if norm_pos:
                    self.gaze_queue.put((norm_pos[0], norm_pos[1]))
            except:
                continue

    def get_latest_gaze(self):
        latest = None
        while not self.gaze_queue.empty():
            latest = self.gaze_queue.get_nowait()
        return latest

    def close(self):
        self.running = False
        if self.listener_thread:
            self.listener_thread.join(timeout=1)
        if self.subscriber:
            self.subscriber.close()
        self.remote.close()
        self.context.term()