from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import pytesseract
from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans
import requests
import os
from dotenv import load_dotenv

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Set your Groq API token
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def decode_image(image_data):
    try:
        if 'base64,' in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        return np.array(Image.open(BytesIO(image_bytes)))
    except Exception as e:
        raise ValueError(f"Image decoding failed: {str(e)}")

def get_dominant_color(image, k=1):
    if image is None or image.size == 0 or len(image.shape) < 3:
        return "rgb(0, 0, 0)"

    pixels = image.reshape(-1, image.shape[-1])
    if pixels.shape[0] < k:
        return "rgb(0, 0, 0)"

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    return f"rgb({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]})"

def estimate_font_size(bbox_height):
    return max(10, min(50, bbox_height // 2))

def identify_element_type(text, width, height):
    text_lower = text.lower()
    if "button" in text_lower or width > 50 and height > 20:
        return "button"
    if "email" in text_lower or "password" in text_lower or "input" in text_lower:
        return "input"
    return "text"

def refine_with_groq(html_code):
    try:
        # Groq API endpoint
        API_URL = "https://api.groq.com/openai/v1/chat/completions"

        # Set headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        # Define the payload with the input HTML code
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "user",
                    "content": f" the given html structure is wrong Enhance this HTML structure by guessing what the Ui might be by using the content in the ui and update it with your own style and design structure with better ui with modern inline CSS with responsiveness to this and give only the html code alone without any descriptions:\n\n{html_code}"
                }
            ]
        }

        # Make the API request
        response = requests.post(API_URL, headers=headers, json=payload)

        # Check response status
        if response.status_code == 200:
            # Extract the generated text from the response
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return f"Failed to refine HTML: {response.text}"
    except Exception as e:
        return f"API Error: {str(e)}"

@app.route("/process", methods=["POST"])
def process_image():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing image data"}), 400
            
        img = decode_image(data["image"])
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        elements = []

        for i in range(len(d['text'])):
            if d['text'][i].strip():
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                if w > 0 and h > 0:
                    font_size = estimate_font_size(h)
                    cropped_img = img[y:y+h, x:x+w] if y+h <= img.shape[0] and x+w <= img.shape[1] else None
                    color = get_dominant_color(cropped_img) if cropped_img is not None and cropped_img.size > 0 else "rgb(0, 0, 0)"
                    element_type = identify_element_type(d['text'][i], w, h)

                    if element_type == "button":
                        elements.append(f"<button style='position: absolute; left: {x}px; top: {y}px; font-size: {font_size}px; background-color: {color}; border: none; padding: 5px 10px;'>{d['text'][i]}</button>")
                    elif element_type == "input":
                        elements.append(f"<input type='text' style='position: absolute; left: {x}px; top: {y}px; width: {w}px; height: {h}px; font-size: {font_size}px; color: {color}; border: 1px solid #ccc; padding: 5px;' placeholder='{d['text'][i]}'>")
                    else:
                        elements.append(f"<div style='position: absolute; left: {x}px; top: {y}px; font-size: {font_size}px; color: {color};'>{d['text'][i]}</div>")

        raw_html = f"<div style='position: relative;'>{''.join(elements)}</div>"
        refined_html = refine_with_groq(raw_html)

        return jsonify({"html": refined_html})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



# 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 


















