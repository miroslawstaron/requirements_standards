import os
import csv
from docx import Document
import requests
import json
import re

def ask_llm(paragraph):
    url = 'http://localhost:11434/api/generate'
    data = {
        "model": "llama3.1:70b",
        "prompt": "You're part of a system that generates requirements about latency based on 3GPP standards. Your objective is to ensure the following paragraph has enough context to become a valid requirement: '{paragraph}'. Answer only using one word, 'POSSIBLE' or 'NO'. 'NO' means that the paragraph is too short (likely not a full sentence), is a heading, a sidenote not containing relevant information or similar." .format(paragraph=paragraph),
        "stream": False
        }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    json_data = json.loads(response.text)
    return json_data['response']

def extract_paragraphs_with_keywords(doc, keywords, filename):
    paragraphs = []
    current_section = ""
    for paragraph in doc.paragraphs:
        if paragraph.style.name.startswith('Heading'):
            current_section = paragraph.text
            continue
        if any(keyword.lower() in paragraph.text.lower() for keyword in keywords):
            if current_section != "" and not any(ignored_section in current_section for ignored_section in ignored_sections):
                paragraphs.append((filename, current_section, paragraph.text))
    return paragraphs

def extract_paragraphs_with_units(doc, unit, filename):
    paragraphs = []
    current_section = ""
    
    # Regex to capture units like "20ms", "20 ms", "20ms."
    unit_regex = re.compile(rf'\b\d+\s*{re.escape(unit)}\b\.?', re.IGNORECASE)
    
    for paragraph in doc.paragraphs:
        if paragraph.style.name.startswith('Heading'):
            current_section = paragraph.text
            continue
        
        # Check if the paragraph contains any units matching the regex pattern
        if re.search(unit_regex, paragraph.text):
            if current_section != "" and not any(ignored_section in current_section for ignored_section in ignored_sections):
                paragraphs.append((filename, current_section, paragraph.text))
    
    return paragraphs

def process_docx_files_in_folder(folder_path, unit, search_word, output_csv, output_units_csv):
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerow(['File', 'Chapter', 'Paragraph', 'LLM response'])
        for filename in os.listdir(folder_path):
            if filename.endswith('.docx'):
                file_path = os.path.join(folder_path, filename)
                print(f"Processing file: {file_path}")
                doc = Document(file_path)
                found_paragraphs = extract_paragraphs_with_keywords(doc, search_word, filename)
                for filename, section, paragraph in found_paragraphs:
                    csvwriter.writerow([filename[:-5], section, paragraph, ask_llm(paragraph)])

    with open(output_units_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerow(['File', 'Chapter', 'Paragraph'])
        for filename in os.listdir(folder_path):
            if filename.endswith('.docx'):
                file_path = os.path.join(folder_path, filename)
                print(f"Processing file: {file_path}")
                doc = Document(file_path)
                found_paragraphs = extract_paragraphs_with_units(doc, unit, filename)
                
                for filename, section, paragraph in found_paragraphs:
                    csvwriter.writerow([filename[:-5], section, paragraph])


folder_path = "standards/22_standards"
keywords = ["latency", "latencies"]
unit = "ms"
ignored_sections = ["References", "Appendix", "Definitions", "Abbreviations"]
output = "outputs/latency_paragraphs.csv"
output_units = "outputs/ms_paragraphs.csv"

process_docx_files_in_folder(folder_path, unit, keywords, output, output_units)
