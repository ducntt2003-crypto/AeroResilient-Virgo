#!/usr/bin/env python3
import os
import cv2
from PIL import Image

def optimize_gif(input_path, output_path, max_width=640, step=6, fps=15):
    print(f"--> Optimizing {input_path} -> {output_path} (max_w={max_width}, step={step}, fps={fps})...")
    cap = cv2.VideoCapture(input_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            h, w = frame.shape[:2]
            if w > max_width:
                new_w = max_width
                new_h = int(h * (max_width / w))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame).quantize(colors=96, method=Image.MEDIANCUT)
            frames.append(pil_img)
        idx += 1
    cap.release()
    
    if len(frames) > 0:
        duration_ms = int(1000 / fps)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=duration_ms,
            loop=0
        )
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"SUCCESS: {output_path} -> {len(frames)} frames, {size_mb:.2f} MB")

if __name__ == "__main__":
    demo_in = "/home/duc/Raptor_UAV/Virgo_RL/Video/Screencast-from-11-19-2026-08_39_32-AM.gif"
    demo_out = "/home/duc/Raptor_UAV/Virgo_RL/docs/media/virgo_figure8_demonstration.gif"
    
    swarm_in = "/home/duc/Raptor_UAV/Virgo_RL/Video/Screencast-from-11-19-2026-08_37_18-AM.gif"
    swarm_out = "/home/duc/Raptor_UAV/Virgo_RL/docs/media/virgo_256_swarm_training.gif"
    
    optimize_gif(demo_in, demo_out, max_width=640, step=8, fps=15)
    optimize_gif(swarm_in, swarm_out, max_width=640, step=4, fps=15)
