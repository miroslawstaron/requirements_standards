import os
import requests
import json
from docx import Document
from docx.shared import Inches
import base64
from PIL import Image

def ask_llm(encoded_image):
    url = f'http://localhost:11435/api/generate'
    data = {
        "model": "minicpm-v",
        "prompt": "Describe this sequence diagram, found in a 3GPP standard. Elicit as much specific information as possible.",
        "images": [encoded_image],
        "stream": False
        }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    json_data = json.loads(response.text)
    return json_data['response']

def extract_images_from_docx(docx_path, output_folder):
    doc = Document(docx_path)
    rels = doc.part.rels
    for rel in rels:
        if "image" in rels[rel].target_ref:
            img = rels[rel].target_part.blob
            img_name = os.path.basename(rels[rel].target_ref)
            img_path = os.path.join(output_folder, img_name)
            with open(img_path, "wb") as img_file:
                img_file.write(img)

def encode_images_to_base64(output_folder):
    for img_name in os.listdir(output_folder):
        img_path = os.path.join(output_folder, img_name)
        if img_name.endswith(".emf"):
            new_name = img_name.replace(".emf", ".png")
            new_path = os.path.join(output_folder, new_name)
            Image.open(img_path).save(output_folder + "/" + new_name)
            with open(new_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                print(f"Finished encoding {new_name}.")

                llm_response = ask_llm(encoded_string)

                txt_path = os.path.join(output_folder, f"{new_name[0:-4]}.txt")
                with open(txt_path, "w") as txt_file:
                    txt_file.write(llm_response)

def main():
    standards_folder = "standards/23502"
    output_folder = "extracted_images"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(standards_folder):
        if filename.endswith(".docx"):
            docx_path = os.path.join(standards_folder, filename)
            extract_images_from_docx(docx_path, output_folder)
            print(f"Extracted images from {filename}")

    encode_images_to_base64(output_folder)
    

if __name__ == "__main__":
    main()