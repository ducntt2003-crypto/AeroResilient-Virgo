#!/usr/bin/env python3
import os
import cv2
from PIL import Image

def optimize_gif(input_path, output_path, max_width=640, step=3, duration_ms=110, start_frame=0, end_frame=None, colors=96):
    print(f"--> Optimizing {input_path} -> {output_path} (max_w={max_width}, step={step}, duration={duration_ms}ms)...")
    cap = cv2.VideoCapture(input_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if end_frame is not None and idx > end_frame:
            break
        if idx >= start_frame and (idx - start_frame) % step == 0:
            h, w = frame.shape[:2]
            if w > max_width:
                new_w = max_width
                new_h = int(h * (max_width / w))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame).quantize(colors=colors, method=Image.MEDIANCUT)
            frames.append(pil_img)
        idx += 1
    cap.release()
    
    if len(frames) > 0:
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
    
    # Demonstration: slow, clear observation (0.9x real-time, duration=110ms, step=3)
    optimize_gif(demo_in, demo_out, max_width=640, step=3, duration_ms=110, start_frame=150, end_frame=1950, colors=96)
    # Swarm: 256 drone training overview
    optimize_gif(swarm_in, swarm_out, max_width=640, step=4, duration_ms=66, colors=96)
