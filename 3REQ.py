import gradio as gr
from docx2python import docx2python
import os

def extractLatencySections(doc):
    strSectionTitle = ""
    dictSections = {}
    listLatency = []

    doc_result = docx2python(doc,
                             paragraph_styles = True, 
                             html=True)

    # we iterate over all lines
    # look for the section titles (which have the tag <h1>, <h2>, <h3>, etc.)
    # then we add the content of each section to the dictionary
    # and if there is a word "latency" somewhere in the section, we add the section title to the listLatency
    for oneLine in doc_result.text.split('\n'):
        print(f'Processing: {oneLine}')
        if "<h" in oneLine:
            strSectionTitle = oneLine
            dictSections[strSectionTitle] = []

        if strSectionTitle != "":  
            dictSections[strSectionTitle].append(oneLine)

        keywordsInLine = ["latency", "latencies"]
        keywordsInSections = ["references", "introduction"]

        if any(word in oneLine.lower() for word in keywordsInLine) and not any(word in strSectionTitle.lower() for word in keywordsInSections): 
            listLatency.append(strSectionTitle)

            
    # remove the keys from the dictionary if they are not part of the listLatency
    # as we want to get only the relevant sections, i.e., the one with the word latency
    for key in list(dictSections.keys()):
        if key not in listLatency:
            del dictSections[key]

    # return the dictionary with the relevant sections
    return dictSections

def process_document(uploaded_file):
    # The uploaded file is provided directly as a path
    filepath = str(uploaded_file)  # Convert NamedString to string if necessary

    # Use the existing function to process the document
    sections_with_latency = extractLatencySections(filepath)

    output_text = sections_with_latency
    print(sections_with_latency.keys)
    
    # Format the output as text or use a DataFrame for nicer formatting
    #output_text = "\n".join([f"Section: {key}\nContent: {sections_with_latency[key]}" for key in sections_with_latency])

    # Optional: Clean up the uploaded file after processing if you don't need to keep it
    os.remove(filepath)

    return output_text

# Create a Gradio interface
iface = gr.Interface(
    fn=process_document,
    inputs=gr.File(label="Upload DOCX File"),
    outputs="text",
    title="3REQ System Interface",
    description="Upload a .docx file to extract sections containing the word 'latency'."
)

# Launch the application
iface.launch()
