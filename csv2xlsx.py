"""
The csv2xlsx converter processes a CSV file with the format (File; Chapter; Paragraph; LLM Response) and generates an XLSX file. 
The given keyword is highlighted in bold, and the data is divided into three separate sheets based on the LLM response 
("YES", "NO", "POSSIBLE"), with appropriate highlighting applied for each state.
"""

import csv
import xlsxwriter
import re

def csv_to_xlsx(input_csv, output_xlsx, keyword):
    print("Processing CSV to XLSX...")

    columns = ["File", "Chapter", "Paragraph", "LLM Response"]
    # Define response types
    responses = ["YES", "NO", "POSSIBLE"]

    # Create a new XLSX file
    workbook = xlsxwriter.Workbook(output_xlsx)

    # Dictionary for storing the worksheets
    sheets = {}

    # Create a sheet for each response type
    for res in responses:
        sheets[res] = workbook.add_worksheet(res)  # Create the sheet
        sheets[res].write_row(0, 0, columns)       # Write headers
        sheets[res].set_column('A:A', 10)  # File column
        sheets[res].set_column('B:B', 18)  # Chapter column
        sheets[res].set_column('C:C', 45)  # Paragraph column
        sheets[res].set_column('D:D', 10)  # LLM Response column

   
    bold_format = workbook.add_format({'bold': True})
    wrapped_format = workbook.add_format({'text_wrap': True})
    yes_format = workbook.add_format({'bold': True, 'bg_color':'#54f542', 'align': 'center'})
    no_format = workbook.add_format({'bold': True, 'bg_color': '#de4343', 'align': 'center'})
    possible_format = workbook.add_format({'bold': True, 'bg_color': '#f7f55e', 'align': 'center'})

    # Compile regex pattern for the keyword with case insensitivity
    keyword_pattern = re.compile(re.escape(keyword), re.IGNORECASE) # CHAT-GPT code

    # Open the CSV file
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            file, chapter, paragraph, llm_response = row
            # Check if the response is one of the defined responses
            if llm_response in responses:
                ws = sheets[llm_response]
                row_idx = ws.dim_rowmax + 1  # Get the next available row (after the header)

                # Write data to the worksheet
                ws.write(row_idx, 0, file)         # Column 0: File
                ws.write(row_idx, 1, chapter)      # Column 1: Chapter

                # Column 3: LLM Response
                if llm_response == "YES":
                    ws.write(row_idx, 3, llm_response, yes_format) 
                elif llm_response == "NO":
                    ws.write(row_idx, 3, llm_response, no_format)
                elif llm_response == "POSSIBLE":
                    ws.write(row_idx, 3, llm_response, possible_format)
                else:
                    ws.write(row_idx, 3, llm_response)

                # Check if the keyword is in the paragraph
                matches = keyword_pattern.finditer(paragraph)  # Use finditer for all matches
                rich_text = []
                start_index = 0

                for match in matches: # Logic of this loop comes from CHAT-GPT
                    # Add the text before the match
                    if match.start() > start_index:
                        rich_text.append(paragraph[start_index:match.start()])  # Normal text
                    # Append the matched keyword with bold format
                    rich_text.append(bold_format)  # Bold format for keyword
                    rich_text.append(match.group(0))  # The actual keyword
                    start_index = match.end()  # Update start_index to end of match

                # Add any remaining text after the last match
                if start_index < len(paragraph):
                    rich_text.append(paragraph[start_index:])  # Normal text

                # Write the rich text back to the 'Paragraph' cell with wrapping
                if rich_text:
                    ws.write_rich_string(row_idx, 2, *rich_text)  # Column 2: Paragraph
                else:
                    # Write paragraph normally with wrapping if the keyword is not found
                    ws.write(row_idx, 2, paragraph, wrapped_format)
                ws.set_row(row_idx, None, wrapped_format)  # Enable text wrapping for the entire row
    # Close the workbook
    workbook.close()
    print(f"Conversion complete! Output saved to '{output_xlsx}'.")

# Example usage
csv_to_xlsx("outputs\latency_paragraphs.csv", "outputs\output.xlsx", "latency")
