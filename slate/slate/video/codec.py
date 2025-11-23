import io
import struct
import zlib


def delta_encode_frames(frames: list[str]) -> bytes:
    """
    Compress video frames using delta-encoding

    Args:
        frames: list of frames to compress
    
    Return:
        byte stream of compressed frames
    """
    frame_stream = io.BytesIO()
    
    prev_frame_bytes = frames[0].encode('utf-8')
    compressed_first = zlib.compress(prev_frame_bytes)
    
    frame_stream.write(struct.pack('<I', len(compressed_first)))
    frame_stream.write(compressed_first)

    for frame in frames[1:]:
        frame_bytes = frame.encode('utf-8')
        
        # delta-encode frame
        max_len = max(len(prev_frame_bytes), len(frame_bytes))
        prev_padded = prev_frame_bytes + b'\x00' * (max_len - len(prev_frame_bytes))
        curr_padded = frame_bytes + b'\x00' * (max_len - len(frame_bytes))
        
        frame_delta = bytes((p - c) % 256 for p, c in zip(prev_padded, curr_padded))
        
        compressed_delta = zlib.compress(frame_delta)
        
        # write frame header
        frame_stream.write(struct.pack('<I', len(frame_bytes)))  # used for decoding
        frame_stream.write(struct.pack('<I', len(compressed_delta)))

        # write frame data
        frame_stream.write(compressed_delta)
        
        prev_frame_bytes = frame_bytes

    return frame_stream.getvalue()


def delta_decode_frames(encoded_data: bytes) -> list[str]:
    """
    Decompress delta-encoded frames

    Args:
        encoded_data: delta-encoded compressed data to decode
    
    Return:
        list of decoded video frames
    """
    frames = []
    stream = io.BytesIO(encoded_data)
    
    length_bytes = stream.read(4)
    compressed_len = struct.unpack('<I', length_bytes)[0]
    compressed_data = stream.read(compressed_len)
    
    prev_frame_bytes = zlib.decompress(compressed_data)
    frames.append(prev_frame_bytes.decode('utf-8'))
    
    while stream.tell() < len(encoded_data):
        # read frame metadata
        frame_len_bytes = stream.read(4)
        frame_length = struct.unpack('<I', frame_len_bytes)[0]
        
        comp_len_bytes = stream.read(4)
        compressed_len = struct.unpack('<I', comp_len_bytes)[0]
        
        compressed_delta = stream.read(compressed_len)
        frame_delta = zlib.decompress(compressed_delta)
        
        # reconstruct frame
        max_len = max(len(prev_frame_bytes), len(frame_delta))
        prev_padded = prev_frame_bytes + b'\x00' * (max_len - len(prev_frame_bytes))
        delta_padded = frame_delta + b'\x00' * (max_len - len(frame_delta))

        frame_bytes = bytes((p - d) % 256 for p, d in zip(prev_padded, delta_padded))
        
        frame_bytes = frame_bytes[:frame_length]  # remove padding
        frames.append(frame_bytes.decode('utf-8'))
        
        prev_frame_bytes = frame_bytes
    
    return frames


def encode_video_to_s4(recording: dict) -> bytes:
    """
    Encode a recording dictionary to S4 format

    Currently does not save: id, timestamp data, per-frame 'info' or 'done' data
    
    Format:
        [magic: 4 bytes]
        [version: 2 bytes]
        [checkpoint_len: 2 bytes]
        [checkpoint: checkpoint_len bytes]
        [action_count: 2 bytes]
        [action_space: variable]
        [frame_count: 2 bytes]
        [encoded_frames: variable]
        [metadata: variable]
    
    Args:
        recording: dictionary containing run information

    Return:
        byte stream containing the run data in the S4 format
    """
    buf = io.BytesIO()
    
    # header
    magic = b'S4V1'
    version = 1
    checkpoint = recording['checkpoint'].encode('utf-8')
    checkpoint_len = len(checkpoint)

    buf.write(magic)
    buf.write(struct.pack('<H', version))
    buf.write(struct.pack('<H', checkpoint_len))
    buf.write(checkpoint)
    
    # action data
    actions = sorted(list(set([m['action'] for m in recording['metadata']])))
    action_to_index = {action: idx for idx, action in enumerate(actions)}
    buf.write(struct.pack('<H', len(actions)))

    for action_name in actions:
        action_name_bytes = action_name.encode('utf-8')
        buf.write(struct.pack('<H', len(action_name_bytes)))
        buf.write(action_name_bytes)

    # frame data
    buf.write(struct.pack('<H', len(recording['frames'])))
    encoded_frames = delta_encode_frames(recording['frames'])
    buf.write(struct.pack('<I', len(encoded_frames)))
    buf.write(encoded_frames)
    
    # frame metadata
    for meta in recording['metadata']:
        # reward data
        reward = int(meta['reward'] * 100)  # truncate and convert to int
        buf.write(struct.pack('<h', reward))

        # action
        action_idx = action_to_index[meta['action']]
        buf.write(struct.pack('<B', action_idx))
        
        # q-values
        q_values = meta.get('q_values', [])
        buf.write(struct.pack('<B', len(q_values)))

        for q_val in q_values:
            q_int = int(round(q_val * 100))
            buf.write(struct.pack('<h', q_int))
    
    return buf.getvalue()


def decode_s4_to_video(s4_data: bytes) -> dict:
    """
    Decode S4 format to a recording dictionary

    Data such as 'done', total_steps, total_reward are not stored in the S4
    format, However these can be reconstructed from the encoded data. 
    
    All frames are set to done=False, with the exception of the last frame 
    which is set to True

    Args:
        s4_data: byte stream of S4 video data

    Return:
        dictionary containing run information from S4 file
    """
    stream = io.BytesIO(s4_data)
    
    # header
    magic = stream.read(4)
    if magic != b'S4V1':
        raise ValueError(f"Invalid magic number: {magic}")
    
    version = struct.unpack('<H', stream.read(2))[0]
    checkpoint_len = struct.unpack('<H', stream.read(2))[0]
    checkpoint = stream.read(checkpoint_len).decode('utf-8')
    
    # read action space
    action_count = struct.unpack('<H', stream.read(2))[0]
    actions = []
    for _ in range(action_count):
        action_len = struct.unpack('<H', stream.read(2))[0]
        action = stream.read(action_len).decode('utf-8')
        actions.append(action)
    
    # frame data
    frame_count = struct.unpack('<H', stream.read(2))[0]
    
    frames_len = struct.unpack('<I', stream.read(4))[0]
    encoded_frames = stream.read(frames_len)
    frames = delta_decode_frames(encoded_frames)
    
    # per-frame metadata
    metadata = []
    for _ in range(frame_count):
        # reward
        reward_int = struct.unpack('<h', stream.read(2))[0]
        reward = reward_int / 100.0
        
        # action
        action_idx = struct.unpack('<B', stream.read(1))[0]
        action = actions[action_idx] if action_idx < len(actions) else 'ERR'
        
        # q_values
        q_count = struct.unpack('<B', stream.read(1))[0]
        q_values = []
        for _ in range(q_count):
            q_int = struct.unpack('<h', stream.read(2))[0]
            q_values.append(q_int / 100.0)
        
        metadata.append({
            'reward': reward,
            'action': action,
            'q_values': q_values,
            'done': False,
            'info': {},
            'timestep': ''
        })
    
    metadata[-1]['done'] = True
    total_reward = sum(m['reward'] for m in metadata)
    
    return {
        'id': None,
        'timestamp': '',
        'total_steps': frame_count,
        'total_reward': total_reward,
        'checkpoint': checkpoint,
        'frames': frames,
        'metadata': metadata
    }