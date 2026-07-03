import os
import base64
import openai
import pandas as pd
import re

client = openai.Client(
    api_key="api-key",
    base_url="openai"
)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_vector(result: str):
    if not result:
        return None

    result = result.strip()

    pattern = r'[01]\s*\(\s*[0-1]?\.?\d+\s*\)'

    matches = re.findall(pattern, result)

    if len(matches) >= 5:
        cleaned_matches = [m.replace(" ", "") for m in matches[:5]]
        return ", ".join(cleaned_matches)
    return None


def analyze_thyroid_nodules(front_image_path, side_image_path):

    if not os.path.exists(front_image_path):
        print(f" No transverse images: {front_image_path}")
        return None
    if not os.path.exists(side_image_path):
        print(f" No Longitudinal images: {side_image_path}")
        return None

    base64_front = encode_image_to_base64(front_image_path)
    base64_side = encode_image_to_base64(side_image_path)

    system_prompt = """You are a professional radiologist. I will send both transverse and longitudinal ultrasound images of the lesion simultaneously, meaning each patient includes two images of the lesion. When analyzing C-TIRADS 4 thyroid nodules on ultrasound, strictly adhere to the following assessment criteria. If either image meets the criteria below, the patient is considered compliant:
Rule 1: Vertical orientation (0/1) When calculating dimensions, if the image contains reference lines, use these lines to determine the length. 0: Horizontal orientation. The anteroposterior diameter of the nodule is less than or equal to the longitudinal diameter in the longitudinal section or the transverse diameter in the transverse section. 1: Vertical orientation. The anteroposterior diameter of the nodule is larger than the longitudinal diameter in the longitudinal section or the transverse diameter in the transverse section.
Rule 2: Solid composition (0/1) 0: Predominately solid, predominately cystic, cystic, or spongiform. 1: Solid composition. The nodule is entirely composed of solid tissue without any cystic components.
Rule 3: Markedly hypoechoic (0/1) 0: Hyperechoic, isoechoic, hypoechoic, or anechoic. 1: Markedly hypoechoic. The echogenicity of the nodule is lower than that of the strap muscles of the neck.
Rule 4: Microcalcifications (0/1) 0: No echogenic foci, comet-tail artifacts, macrocalcifications, or peripheral calcifications without definite microcalcifications.1: Microcalcifications, defined as punctate echogenic foci of less than about 1 mm with or without shadowing. When different types of punctate echogenic foci are present simultaneously, record microcalcifications preferentially according to the principle of suspicious feature priority.
Rule 5: ill-defined or irregular margins, or extrathyroidal extension (0/1) 0: Circumscribed margin, appearing as a clear, smooth, and complete curve. 1: Ill-defined margin, irregular margin, or extrathyroidal extension. Irregular margin includes spiculated, angular, or microlobulated margins. 
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
        print(f" API call failed: {e}")
        return None


def process_csv(csv_path, output_csv="result.csv"):


    df = pd.read_csv(csv_path)

    if "image1_path" not in df.columns or "image2_path" not in df.columns:
        raise ValueError("CSV must have image1_path and image2_path")

    vectors = []
    for i, row in df.iterrows():
        img1 = row["image1_path"]
        img2 = row["image2_path"]

        print(f" Analyzing the {i + 1}/{len(df)} : {img1} | {img2}")
        vector = analyze_thyroid_nodules(img1, img2)

        if vector:
            print(f"   -> Result: {vector}")
            vectors.append(vector)
        else:
            print(f"   -> Extraction failed")
            vectors.append("ERROR")

    df["vector_result"] = vectors
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n The analysis is complete, and the results have been saved to: {output_csv}")


if __name__ == "__main__":
    process_csv("data/train.csv", "data/result_gpt5.2-2.csv")