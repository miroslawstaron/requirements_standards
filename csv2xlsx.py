"""
The csv2xlsx converter processes a CSV file with the format (File; Chapter; Paragraph; LLM Response) and generates an XLSX file. 
The given keyword is highlighted in bold, and the data is divided into three separate sheets based on the LLM response 
("YES", "NO", "POSSIBLE"), with appropriate highlighting applied for each state.
"""

import csv
import xlsxwriter
import re

# TODO: Add keyword highlighting and better formatting for readability

def csv_to_xlsx(input_csv, output_xlsx, keyword):
    print("Processing CSV to XLSX...")

    print(input_csv[0])
    print(input_csv[1])
    print(input_csv[2])

    # different input files
    # 1 and 2 have the same structure so different logic is only needed for 0
    new_requirements = input_csv[0]; # "outputs\new_requirements.csv"
    latency_paragraphs = input_csv[1]; # "outputs\latency_paragraphs.csv"
    latency_no_paragraphs = input_csv[2]; # "outputs\latency_no_paragraphs.csv"

    # Create a new XLSX file
    workbook = xlsxwriter.Workbook(output_xlsx)

    # Create a sheet for each input file
    workbook.add_worksheet("new_requirements")
    workbook.add_worksheet("latency_paragraphs")
    workbook.add_worksheet("latency_no_paragraphs")

    # Create columns for the sheets
    for sheet in workbook.worksheets():
        sheet.write_row(0, 0, ["File", "Chapter", "Paragraph", "LLM Response"]) # headers
        sheet.set_column('A:A', 10)  # File column
        sheet.set_column('B:B', 18)  # Chapter column
        sheet.set_column('C:C', 45)  # Paragraph column
        sheet.set_column('D:D', 10)  # LLM Response column
        if sheet.get_name() == "new_requirements":
            sheet.write(0, 4, "Requirement") # Requirement column
            sheet.set_column('E:E', 45)  # Requirement column

    # read new_requirements.csv and write to its sheet
    with open(new_requirements, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            file, chapter, paragraph, llm_response, requirement = row
            sheet = workbook.get_worksheet_by_name("new_requirements") # Get the sheet
            row_idx = sheet.dim_rowmax + 1 # Get the next available row (after the header)
            sheet.write(row_idx, 0, file)         # Column 0: File
            sheet.write(row_idx, 1, chapter)      # Column 1: Chapter
            sheet.write(row_idx, 2, paragraph)    # Column 2: Paragraph
            sheet.write(row_idx, 3, llm_response) # Column 3: LLM Response
            sheet.write(row_idx, 4, requirement) # Column 4: Requirement
    
    # read latency_paragraphs.csv and write to its sheet
    with open(latency_paragraphs, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            file, chapter, paragraph, llm_response = row
            sheet = workbook.get_worksheet_by_name("latency_paragraphs")
            row_idx = sheet.dim_rowmax + 1 # Get the next available row (after the header)
            sheet.write(row_idx, 0, file)         # Column 0: File
            sheet.write(row_idx, 1, chapter)      # Column 1: Chapter
            sheet.write(row_idx, 2, paragraph)    # Column 2: Paragraph
            sheet.write(row_idx, 3, llm_response) # Column 3: LLM Response

    # read latency_no_paragraphs.csv and write to its sheet
    with open(latency_no_paragraphs, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            file, chapter, paragraph, llm_response = row
            sheet = workbook.get_worksheet_by_name("latency_no_paragraphs")   
            row_idx = sheet.dim_rowmax + 1 # Get the next available row (after the header)     
            sheet.write(row_idx, 0, file)         # Column 0: File
            sheet.write(row_idx, 1, chapter)      # Column 1: Chapter
            sheet.write(row_idx, 2, paragraph)    # Column 2: Paragraph
            sheet.write(row_idx, 3, llm_response) # Column 3: LLM Response

    #close the workbook
    workbook.close()
    print(f"Conversion complete! Output saved to '{output_xlsx}'.")
   


# Example usage
outputs = ['outputs\\new_requirements.csv', "outputs\latency_paragraphs.csv", "outputs\latency_no_paragraphs.csv"]

csv_to_xlsx(outputs, "outputs\output.xlsx", "latency")


