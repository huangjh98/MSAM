import os
import cv2
import json
import time
import PIL.Image
import google.generativeai as genai

# 1. API 配置
genai.configure(api_key='你的_API_KEY')
model = genai.GenerativeModel('gemini-xxx')


PROMPT_A = """As a professional remote sensing and aerial image captioning system, you generate detailed descriptions based on the given image. The annotation process follows the guidelines below:
Principle 1: The description first states the fundamental attributes of the image, including its acquisition method (satellite or aerial), imaging type (color or panchromatic), and resolution level, establishing the basis for overall understanding.
Principle 2: Descriptions of objects in the image rely solely on observable facts, covering their quantity, color, material, shape, size, and their absolute and relative positions within the frame. No inferred content is allowed.
Principle 3: The annotation process begins with an overview of the entire scene and then transitions to specific objects to maintain a clear and coherent structure.
Principle 4: All descriptions strictly depend on verifiable visual information and avoid subjective speculation. The narrative uses natural, continuous language and avoids list-like formatting and unnecessary decorative expressions."""

VIDEO_DIR = "./videos/"           # 原始视频存放路径
SAVE_FRAME_DIR = "./sampled_frames/" # 采样帧临时存放路径
INTERMEDIATE_FILE = "./frame_descriptions.json"

def extract_frames(video_path, save_dir, k):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_id = os.path.basename(video_path).split('.')[0]
    
    indices = [int(i * (total_frames - 1) / 11) for i in range(k)]
    
    frame_paths = []
    current_idx = 0
    success, frame = cap.read()
    
    target_dir = os.path.join(save_dir, video_id)
    os.makedirs(target_dir, exist_ok=True)

    while success:
        if current_idx in indices:
            p = os.path.join(target_dir, f"frame_{current_idx}.jpg")
            cv2.imwrite(p, frame)
            frame_paths.append(p)
        current_idx += 1
        success, frame = cap.read()
    
    cap.release()
    return frame_paths

def run_stage_1():
    results = {}
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

    for v_file in video_files:
        v_id = v_file.split('.')[0]
        print(f"正在处理视频: {v_id}")
        
        # 1. 切帧
        frame_paths = extract_frames(os.path.join(VIDEO_DIR, v_file), SAVE_FRAME_DIR, k=5)
        
        # 2. 调用 Prompt A 生成描述
        caps = []
        for p in frame_paths:
            try:
                img = PIL.Image.open(p)
                response = model.generate_content([PROMPT_A, img])
                caps.append(response.text.strip())
                time.sleep(1) # API 频率限制
            except Exception as e:
                print(f"标注失败 {p}: {e}")
        
        results[v_id] = caps
        
        # 保存中间结果
        with open(INTERMEDIATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    run_stage_1()
