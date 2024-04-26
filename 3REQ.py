import gradio as gr
from docx2python import docx2python
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
import io
from PIL import Image

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def extractEmbeddings(lstAllLines):
    '''
    This function extracts the embeddings for the sections of the document.
    It uses the SentenceTransformer model to encode the sentences in the sections.
    The embeddings are then averaged to get the embedding for the whole section.
    '''

    # list with all embeddings for the sections
    lstEmbeddings = []

    for oneLine in lstAllLines:

        # the content of the section starts on the third position of the list
        sentences = oneLine[3:]

        # Sentences are encoded by calling model.encode()
        embeddings = model.encode(sentences)
        
        lstOneLine = [oneLine[0], oneLine[1], 2, str(sentences).replace("$", "_").replace("\n", "_"), embeddings]

        lstEmbeddings.append(lstOneLine)
    
    return lstEmbeddings

def embedRefereces():
    # open the file List.xlsx using pandas
    # and read the workshop NR
    df = pd.read_excel("./List.xlsx", sheet_name="R_NR")

    # convert the dataframe to a list of lists
    lstReference = df.values.tolist()

    # list with all embeddings for the sections
    lstEmbeddingsRef = []

    for oneLine in lstReference:

        # the content of the section starts on the third position of the list
        sentences = oneLine[0]

        # Sentences are encoded by calling model.encode()
        embeddings = model.encode(sentences)
        
        # Print the average embeddings for all the sentences 
        # in this section
        avg_embedding = embeddings
        
        lstOneLine = [oneLine[0], 'REF', oneLine[1], oneLine[1], avg_embedding]

        lstEmbeddingsRef.append(lstOneLine)

    # return all embeddings
    return lstEmbeddingsRef

def filterRelevant(lstEmbeddings, lstEmbeddingsRef):
    # for each line in lstEmbeddings
    # we calculate the euclidean distance with each line in lstEmbeddingsRef

    lstDistPos = []
    lstDistNeg = []

    lstRelevant = []

    for oneLine in lstEmbeddings:    
        for oneLineRef in lstEmbeddingsRef:

            print(f"Processing: {oneLine} with {oneLineRef}")

            if oneLineRef[2] == 1:
                # euclidean distance between the two embeddings
                dist = euclidean_distances([oneLine[4]], [oneLineRef[4]])
                lstDistPos.append(dist[0][0])
            if oneLineRef[2] == 0:
                # euclidean distance between the two embeddings
                dist = euclidean_distances([oneLine[4]], [oneLineRef[4]])
                lstDistNeg.append(dist[0][0])
        
        # now calculate the average for both lists
        avgDistPos = np.mean(lstDistPos)
        avgDistNeg = np.mean(lstDistNeg)

        if avgDistPos < avgDistNeg:
            #print(f"Section {oneLine[0]} is relevant")
            # add the class to the list
            oneLine.append(1)
            lstRelevant.append(oneLine)
        else:
            #print(f"Section {oneLine[0]} is not relevant")
            # add the class to the list
            oneLine.append(0)

    return lstRelevant

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
        
        if "<h" in oneLine:
            strSectionTitle = oneLine
            dictSections[strSectionTitle] = []
            #print(f'Processing: {oneLine}')

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


def visualize(lstEmbeddingsAll):
    # let's plot these average embeddings using t-SNE

    # we create a list with the embeddings
    lstEmbeddingsNP = np.array([x[4] for x in lstEmbeddingsAll])

    # we use t-SNE to reduce the dimensionality of the embeddings
    tsne = TSNE(n_components=2, verbose=0, perplexity=10, n_iter=300)
    tsne_results = tsne.fit_transform(lstEmbeddingsNP)

    # Create a color map based on x[2]
    color_map = {0: 'red', 1: 'green', 2: 'blue'}
    colors = [color_map[x[2]] for x in lstEmbeddingsAll]

    # we plot the t-SNE results
    plt.figure(figsize=(16,10))

    plt.scatter(tsne_results[:,0], tsne_results[:,1],c=colors, s=100, alpha=0.5)

    # Add labels to each dot
    for i, label in enumerate([x[0] for x in lstEmbeddingsAll]):
        plt.text(tsne_results[i, 0], tsne_results[i, 1], label[:10]+'...')

    # Convert the matplotlib figure to a PIL Image and then to a numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)

    return img_array


# this is the main function that processes the document and returns the relevant sections
def process_document(uploaded_file):

    strOutput = ""

    # The uploaded file is provided directly as a path
    filepath = str(uploaded_file)  

    # Use the existing function to process the document
    sections_with_latency = extractLatencySections(filepath)

    # extract embeddings
    lstEmbeddings = extractEmbeddings(sections_with_latency)

    # embed references
    lstEmbeddingsRef = embedRefereces()

    # filter relevant sections
    lstRelevant = filterRelevant(lstEmbeddings, lstEmbeddingsRef)

    # Format the output as text or use a DataFrame for nicer formatting
    strOutput = strOutput + "\n".join([f"Section: {key}\n" for key in sections_with_latency])

    visRelevant = visualize(lstRelevant)

    # if the string is empty then write "no sections found", 
    # otherwise write the section to the output
    if strOutput == "":
        strOutput = "No sections found"
    
    # Optional: Clean up the uploaded file after processing if you don't need to keep it
    os.remove(filepath)

    return visRelevant, strOutput

model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a Gradio interface
iface = gr.Interface(
    fn=process_document,
    inputs=gr.File(label="Upload DOCX File"),
    outputs=[gr.Image(), gr.Text()],
    title="3REQ System Interface",
    description="Upload a .docx file to extract sections containing the word 'latency'."
)

# Launch the application
iface.launch()
