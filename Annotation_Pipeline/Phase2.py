import json
import time
import google.generativeai as genai

genai.configure(api_key='你的_API_KEY')
model = genai.GenerativeModel('gemini-xxx-pro')


PROMPT_B_TEMPLATE = """After reviewing and logically integrating the descriptions of the five key frames, write a five-sentence summary. Each sentence presents a coherent overview of the video and highlights key details. Follow these requirements:
Principle 1: Each sentence is approximately 20 words, forming a coherent five-sentence summary.
Principle 2: The description covers main subjects, actions, scenes, and notable features, clearly showing visible changes in composition or timeline.
Principle 3: Include only objective information directly observable from the frames, excluding uncertain or inferred content.
Principle 4: Use concise and objective language, avoid any decorative or subjective expressions, and start descriptions directly. 
Output in this format: 1. xxx 2. xxx 3. xxx 4. xxx 5. xxx

Here are the frame descriptions:
{frame_descriptions}"""

INTERMEDIATE_FILE = "./frame_descriptions.json"
FINAL_LABEL_FILE = "./msam_final_labels.txt"

def run_stage_2():
    with open(INTERMEDIATE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(FINAL_LABEL_FILE, 'a', encoding='utf-8') as out_f:
        for video_id, caps in data.items():
            print(f"生成视频总结: {video_id}")
            
            caps_combined = "\n".join([f"Frame {i+1}: {c}" for i, c in enumerate(caps)])
            prompt = PROMPT_B_TEMPLATE.format(frame_descriptions=caps_combined)
            
            try:
                response = model.generate_content(prompt)
                summary = response.text.strip().replace('\n', ' ')
                out_f.write(f"{video_id} {summary}\n")
                time.sleep(2)
            except Exception as e:
                print(f"总结失败 {video_id}: {e}")

if __name__ == "__main__":
    run_stage_2()
