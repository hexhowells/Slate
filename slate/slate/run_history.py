from collections import deque
from datetime import datetime


class Recording:
    def __init__(self, uuid, data):
        self.run_start_time = datetime.now()
        self.checkpoint = data['checkpoint']
        self.frames = []
        self.total_reward = 0.0
        self.run_id = uuid
        self.metadata = []

        self.add_frame(data)
    

    def add_frame(self, data):
        self.frames.append(data['frame'])
        self.total_reward += max(0, data['reward'])
        self.metadata.append({
            'reward': data['reward'],
            'done': data['done'],
            'info': data['info'],
            'q_values': data['q_values'],
            'action': data['action'],
            'timestep': datetime.now().isoformat()
        })


    def get_recording(self):
        return {
            'id': self.run_id,
            'timestamp': self.run_start_time.isoformat(),
            'total_steps': len(self.frames),
            'total_reward': self.total_reward,
            'checkpoint': self.checkpoint,
            'frames': self.frames,
            'metadata': self.metadata
        }
    

class RunHistory:
    def __init__(self, max_history_size: int=5) -> None:
        """
        Args:
            max_history_size: maximum number of records in the history buffer
        """
        self.max_history_size = max_history_size
        self.run_history = deque(maxlen=max_history_size)
        self.current_recording = None
        self.recording_ids = []
        self.recording_num = 1
    

    @property
    def recording(self):
        return self.current_recording is not None
    

    def check_id(self, uuid):
        return uuid in self.recording_ids
    

    def fetch_recording(self, uuid) -> Recording:
        return self.run_history[(uuid-1) % self.max_history_size]
    

    def fetch_recording_frame(self, uuid, frame):
        run = self.run_history[(uuid-1) % self.max_history_size]
        frames = run.get('frames', [])
        metadata = run.get('metadata', [])
        
        if 0 <= frame < len(frames):
            frame_metadata = metadata[frame] if frame < len(metadata) else {}
            return {
                'frame': frames[frame],
                'reward': frame_metadata.get('reward', 0.0),
                'done': frame_metadata.get('done', False),
                'info': frame_metadata.get('info', {}),
                'q_values': frame_metadata.get('q_values', []),
                'action': frame_metadata.get('action', ''),
                'checkpoint': run.get('checkpoint', '')
            }
        else:
            if len(frames) > 0:
                first_metadata = metadata[0] if len(metadata) > 0 else {}
                return {
                    'frame': frames[0],
                    'reward': first_metadata.get('reward', 0.0),
                    'done': first_metadata.get('done', False),
                    'info': first_metadata.get('info', {}),
                    'q_values': first_metadata.get('q_values', []),
                    'action': first_metadata.get('action', ''),
                    'checkpoint': run.get('checkpoint', '')
                }
            return None
    

    def new_recording(self, data):
        self.current_recording = Recording(self.recording_num, data)
        self.recording_ids.append(self.recording_num)
        self.recording_num += 1


    def update_recording(self, data):
        assert self.current_recording, "Cannot call update_recording - No current recording setup in RunHistory"
        self.current_recording.add_frame(data)


    def stop_recording(self):
        assert self.current_recording, "Cannot call stopc_recording - No current recording setup in RunHistory"
        self.run_history.append(self.current_recording.get_recording())
        self.current_recording = None

    
    def get_run_history(self):
        return list(self.run_history)
    

    def get_history_metadata(self):
        metadata = []
        for run in self.run_history:
            metadata.append({
                'timestamp': run['timestamp'],
                'id': run['id'],
                'total_steps': run['total_steps'],
                'total_reward': run['total_reward'],
            })
        
        return metadata
