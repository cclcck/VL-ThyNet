import os
import base64
import openai
import pandas as pd
import re

# 初始化OpenAI API客户端
client = openai.Client(
    api_key="api-key",
    base_url="openai"
)


def encode_image_to_base64(image_path):
    """将图片文件转换为base64编码字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_vector(result: str):
    """
    提取带置信度的向量格式
    目标格式: 0(0.9), 1(0.85), 0(0.95), 1(0.7), 1(0.9)
    """
    if not result:
        return None

    result = result.strip()

    # 正则表达式解释：
    # [01]          -> 匹配分类结果 0 或 1
    # \s* -> 允许可能有空格
    # \(            -> 匹配左括号
    # \s* -> 允许括号内有空格
    # [0-1]?\.?\d+  -> 匹配浮点数 (如 0.9, .95, 1.0)
    # \s* -> 允许空格
    # \)            -> 匹配右括号
    pattern = r'[01]\s*\(\s*[0-1]?\.?\d+\s*\)'

    # 查找所有符合 "分类(置信度)" 模式的子串
    matches = re.findall(pattern, result)

    # 如果找到了至少5个符合格式的项，取前5个并用逗号连接
    if len(matches) >= 5:
        # 清理一下格式，去掉多余空格，确保格式统一 (例如去掉括号内的空格)
        cleaned_matches = [m.replace(" ", "") for m in matches[:5]]
        return ", ".join(cleaned_matches)

    # 如果没匹配到标准格式，尝试兜底逻辑（可选，视需求而定）
    return None


def analyze_thyroid_nodules(front_image_path, side_image_path):
    """分析甲状腺结节，返回带置信度的向量编码"""

    if not os.path.exists(front_image_path):
        print(f" 正面照不存在: {front_image_path}")
        return None
    if not os.path.exists(side_image_path):
        print(f" 侧面照不存在: {side_image_path}")
        return None

    base64_front = encode_image_to_base64(front_image_path)
    base64_side = encode_image_to_base64(side_image_path)

    system_prompt = """You are a professional radiologist. I will send both anteroposterior and lateral images of the lesion simultaneously—meaning each patient includes two images of the lesion. When analyzing TI-RADS 4 thyroid nodules on ultrasound, strictly adhere to the following assessment criteria. If either image meets the criteria below, the patient is considered compliant:
Rule 1: Morphological Assessment (0/1)
When calculating dimensions, if the image contains reference lines, use these lines to determine anteroposterior (AP) and transverse (T) diameters. The vertical reference line defines AP diameter, and the horizontal reference line defines T diameter. Calculate the aspect ratio using the formula:
0: <1 (width > height)
1: ≥1 (height > width or approximately 1 counts as 1)

Rule 2: Calcification/Hypoechoic Points (0/1)
0: No calcification/Comet-tail sign/Large calcification
1: Microcalcification/Punctate hypoechoic points

Rule 3: Margins (0/1)
0: Smooth or blurred
1: Jagged/Nodular/Irregular margins

Rule 4: Echo Pattern (0/1)
Determined by the nodule's lowest echo region
1: Hypoechoic/Extremely hypoechoic
0: Isoechoic/Hyperechoic

Rule 5: Morphology/Internal Structure (0/1) 0: Cystic/Spongy 1: Solid/Composite solid

Output Format: Output vector, each element containing a binary classification (0 or 1) and a confidence score in parentheses (0.0 to 1.0). Format: Class(Confidence), Class(Confidence), ... Example: 0(0.9), 1(0.85), 0(0.95), 1(0.7), 1(0.9)

Do not include any explanatory text or descriptive content. Provide only vector data with confidence scores."""

    try:
        response = client.chat.completions.create(
            model="gpt-5.2-2025-12-11",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": "Analyze thyroid ultrasound images, outputting only vector and confidence results."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_front}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_side}"}}
                ]}
            ],

            max_completion_tokens=150,
            temperature=0.0,
            stream=False
        )

        result = response.choices[0].message.content.strip()

        vector = extract_vector(result)
        return vector if vector else None

    except Exception as e:
        print(f" API调用失败: {e}")
        # ⚠️ 如果你希望 GUI 能在弹窗/文本框里直接显示 API 的报错，
        # 可以将下一行改为：raise Exception(f"API调用失败: {str(e)}")
        return None


def process_csv(csv_path, output_csv="result.csv"):
    """从 CSV 批量读取 image1_path 和 image2_path，并生成向量结果"""

    df = pd.read_csv(csv_path)

    if "image1_path" not in df.columns or "image2_path" not in df.columns:
        raise ValueError("CSV 必须包含 image1_path 和 image2_path 两列")

    vectors = []
    for i, row in df.iterrows():
        img1 = row["image1_path"]
        img2 = row["image2_path"]

        print(f" 正在分析第 {i + 1}/{len(df)} 条: {img1} | {img2}")
        vector = analyze_thyroid_nodules(img1, img2)

        if vector:
            print(f"   -> 结果: {vector}")
            vectors.append(vector)
        else:
            print(f"   -> 提取失败")
            vectors.append("ERROR")

    df["vector_result"] = vectors
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n 批量分析完成，结果已保存到: {output_csv}")


if __name__ == "__main__":
    process_csv("data/train.csv", "data/result_gpt5.2-2.csv")