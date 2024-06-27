import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import librosa
from model import IRFD, SPEAK
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

class SpeakInference:
    def __init__(self, config, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Initialize models
        self.irfd = IRFD().to(self.device)
        self.speak = SPEAK().to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.irfd.load_state_dict(checkpoint['irfd_state_dict'])
        self.speak.load_state_dict(checkpoint['speak_state_dict'])
        
        self.irfd.eval()
        self.speak.eval()
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.model.sample_size, config.model.sample_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Initialize emotion recognizer
        self.fer = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl')

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def preprocess_audio(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=16000)
        return torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

    def preprocess_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        return torch.stack(frames).to(self.device)

    def generate_talking_head(self, identity_image_path, audio_path, pose_video_path, emotion_video_path):
        with torch.no_grad():
            # Preprocess inputs
            identity_image = self.preprocess_image(identity_image_path)
            audio = self.preprocess_audio(audio_path)
            pose_video = self.preprocess_video(pose_video_path)
            emotion_video = self.preprocess_video(emotion_video_path)

            # Extract features using IRFD
            fi = self.irfd.Ei(identity_image)
            fp = self.irfd.Ep(pose_video)
            fe = self.irfd.Ee(emotion_video)

            # Generate talking head using SPEAK
            generated_frames = self.speak(fi, fe, fp, audio)

            return generated_frames

    def save_video(self, frames, output_path, fps=30):
        frames = frames.cpu().numpy()
        frames = (frames * 255).astype(np.uint8)
        frames = frames.transpose(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames.shape[2], frames.shape[1]))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

def main():
    config = {
        "model": {
            "sample_size": 256
        }
    }
    checkpoint_path = "./speak/checkpoint-5500.pth"
    
    inference = SpeakInference(config, checkpoint_path)
    
    identity_image_path = "path/to/identity_image.jpg"
    audio_path = "path/to/audio.wav"
    pose_video_path = "path/to/pose_video.mp4"
    emotion_video_path = "path/to/emotion_video.mp4"
    output_path = "path/to/output_video.mp4"
    
    generated_frames = inference.generate_talking_head(identity_image_path, audio_path, pose_video_path, emotion_video_path)
    inference.save_video(generated_frames, output_path)
    
    print(f"Generated talking head video saved to {output_path}")

if __name__ == "__main__":
    main()