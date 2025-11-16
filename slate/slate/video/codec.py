"""
S4 file format (Slate/MP4)

header: 
 - env
 - checkpoint
 - action_space (list of action names in order)
 - start time
 - total reward

frames: delta-encoded frames

metadata: metadata for each frame:
 - q-values: truncated to 2dp
 - action: index of action string (in header)
 - reward: integer value


[ header id ] (4 bytes)
[ version ] (2 bytes)
[ checkpoint_len ] (2 bytes)
[ checkpoint ] (checkpoint len)
[ action_space ] (2 bytes)
[ frame_count ] (2 bytes)

"""
import io
import struct
import zlib

"""
Raw size: 135,508
Compressed size: 67,002

Raw size: 128,528
Compressed size: 20,106


"""

def delta_encode_frames(frames) -> bytes:
    """
    take initial frame as-is
    for frame in frames[1:]:
        new_frame = prev_frame - frame
        comp_frame = zlib.compress(new_frame)
    """
    raw_size = 0
    compressed_size = 0
    prev_frame = bytes(frames[0], 'utf-8')
    compressed_size += len(zlib.compress(prev_frame))

    for frame in frames[1:]:
        raw_size += len(frame)

        frame_bytes = bytes(frame, 'utf-8')
        frame_delta = bytes((p - f) % 256 for p, f in zip(prev_frame, frame_bytes))
        compressed_size += len(zlib.compress(frame_delta))

        prev_frame = frame_bytes
    
    print(f'Raw size: {raw_size}\nCompressed size: {compressed_size}')


def encode_video_to_s4(recording) -> bytes:
    buf = io.BytesIO()
    magic = b'S4V1'
    version = 0.1
    checkpoint_len = len(recording['checkpoint'])
    actions = sorted(list(set([m['action'] for m in recording['metadata']])))
    action_space = len(actions)
    frame_count = len(recording['frames'])

    header_struct = struct.Struct("<4sHHH")

    delta_encode_frames(recording['frames'])
    
    # recording['id']
    # recording['timestamp']
    # recording['total_steps']
    # recording['total_reward']
    # recording['checkpoint']
    # recording['frames']
    # recording['metadata']
    return b'0'


def decode_s4_to_video(s4_data):
    pass