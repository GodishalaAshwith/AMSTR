"""
Generate a mini demonstration LMDB dataset for SVTRv2X training & evaluation.
This allows immediately verifying the entire training and evaluation loop on Google Colab
without waiting to download gigabytes of external data.
"""

import os
import io
import lmdb
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_sample_lmdb(output_path, num_samples=50):
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=1024 * 1024 * 100) # 100 MB map size
    cache = {}
    
    sample_texts = [
        "中国智造", "深度学习", "文字识别", "人工智能", "开源视觉",
        "计算机视觉", "模式识别", "图像处理", "端到端识别", "自蒸馏模型",
        "混合专家", "注意力机制", "场景文本", "目标检测", "神经网络",
        "SVTRv2X", "OpenOCR", "Transformer", "AI2026", "OCR"
    ]
    
    for cnt in range(1, num_samples + 1):
        # Create a synthetic text image (H=32, W=128)
        img = Image.new('RGB', (128, 32), color=(np.random.randint(200, 255), np.random.randint(200, 255), np.random.randint(200, 255)))
        draw = ImageDraw.Draw(img)
        text = sample_texts[(cnt - 1) % len(sample_texts)]
        
        # Draw simple text
        draw.text((10, 8), text, fill=(np.random.randint(0, 50), np.random.randint(0, 50), np.random.randint(0, 50)))
        
        # Convert to bytes
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        imageBin = buf.getvalue()
        w, h = img.size
        
        # Store according to LMDB OCR convention
        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        whKey = 'wh-%09d'.encode() % cnt
        
        cache[imageKey] = imageBin
        cache[labelKey] = text.encode('utf-8')
        cache[whKey] = f"{w}_{h}".encode()

    cache['num-samples'.encode()] = str(num_samples).encode()
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)
    env.close()
    print(f"[+] Created {num_samples} samples in: {output_path}")

def main():
    base_dir = "benchmark_dataset"
    splits = [
        ("document/document_train", 100),
        ("document/document_test", 25),
        ("scene/scene_train", 100),
        ("scene/scene_test", 25),
        ("web/web_train", 100),
        ("web/web_test", 25),
        ("handwriting_hwdb_train", 100),
        ("handwriting_ic13_test", 25),
    ]
    
    print("Generating demo LMDB datasets...")
    for rel_path, n in splits:
        full_path = os.path.join(base_dir, rel_path)
        generate_sample_lmdb(full_path, num_samples=n)
    print("\n[SUCCESS] All benchmark demo datasets created successfully!")

if __name__ == "__main__":
    main()

