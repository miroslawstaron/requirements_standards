import os
import csv
from docx import Document

def extract_paragraphs_with_word(doc, word, filename):
    paragraphs = []
    current_section = ""
    for paragraph in doc.paragraphs:
        if paragraph.style.name.startswith('Heading'):
            current_section = paragraph.text
        if word.lower() in paragraph.text.lower():
            paragraphs.append((filename, current_section, paragraph.text))
    return paragraphs


def process_docx_files_in_folder(folder_path, search_word, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerow(['File', 'Section', 'Paragraph'])
        for filename in os.listdir(folder_path):
            if filename.endswith('.docx'):
                file_path = os.path.join(folder_path, filename)
                print(f"Processing file: {file_path}")
                doc = Document(file_path)
                found_paragraphs = extract_paragraphs_with_word(doc, search_word, filename)
                for filename, section, paragraph in found_paragraphs:
                    csvwriter.writerow([filename, section, paragraph])


folder_path = "standards/23_standards"
keyword = "latency"
output = "latency_paragraphs.csv"

process_docx_files_in_folder(folder_path, keyword, output)
