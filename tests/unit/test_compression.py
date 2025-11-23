import pickle
import os
import unittest

from slate.video.codec import encode_video_to_s4, decode_s4_to_video, delta_encode_frames, delta_decode_frames


class TestCompression(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load test data once for all tests"""
        test_file_path = os.path.join(os.path.dirname(__file__), 'test_file.pkl')
        
        with open(test_file_path, 'rb') as f:
            cls.test_dict = pickle.load(f)
        
        cls.test_frames = cls.test_dict['frames']
    

    def test_encode_decode_video(self):
        """
        Test encode_video_to_s4 and decode_s4_to_video round-trip
        
        Flow: dict -> encode -> encoded_data -> decode -> decoded_dict
        
        Assert:
            'total_steps' is equal
            'total_reward' is equal
            'checkpoint' is equal
            frame data is equal
            reward metadata is equal
            action metadata is equal
            q-value metadata is equal
        """
        # dict -> encode -> encoded_data
        encoded_data = encode_video_to_s4(self.test_dict)
        
        # encoded_data -> decode -> decoded_dict
        decoded_dict = decode_s4_to_video(encoded_data)
        
        self.assertEqual(
            self.test_dict['total_steps'], 
            decoded_dict['total_steps'], 
            "Decoded 'total_steps' does not match original data"
        )

        self.assertEqual(
            self.test_dict['total_reward'], 
            decoded_dict['total_reward'], 
            "Decoded 'total_reward' does not match original data"
        )

        self.assertEqual(
            self.test_dict['checkpoint'], 
            decoded_dict['checkpoint'], 
            "Decoded 'checkpoint' does not match original data"
        )

        self.assertEqual(
            self.test_dict['frames'], 
            decoded_dict['frames'], 
            "Decoded frames does not match original data"
        )

        self.assertEqual(
            [frame['reward'] for frame in self.test_dict['metadata']], 
            [frame['reward'] for frame in decoded_dict['metadata']], 
            "Decoded reward metadata does not match original data"
        )

        self.assertEqual(
            [frame['action'] for frame in self.test_dict['metadata']], 
            [frame['action'] for frame in decoded_dict['metadata']], 
            "Decoded action metadata does not match original data"
        )

        self.assertEqual(
            [[round(q, 2) for q in frame['q_values']] for frame in self.test_dict['metadata']], 
            [frame['q_values'] for frame in decoded_dict['metadata']], 
            "Decoded q-value metadata does not match original data"
        )
    
    
    def test_delta_encode_decode_frames(self):
        """
        Test delta_encode_frames and delta_decode_frames round-trip
        
        Extracts the frame data from a pickled run and performs the compression
        and decompression process.

        Flow: frames -> encode -> encoded_data -> decode -> decoded_frames
        
        Assert: 
            frames == decoded_frames
        """
        original_frames = self.test_frames.copy()
        
        # frames -> encode -> encoded_data
        encoded_data = delta_encode_frames(original_frames)
        
        # encoded_data -> decode -> decoded_frames
        decoded_frames = delta_decode_frames(encoded_data)
        
        self.assertEqual(
            original_frames, 
            decoded_frames,
            "Decoded frames should match original frames")


if __name__ == '__main__':
    unittest.main()

