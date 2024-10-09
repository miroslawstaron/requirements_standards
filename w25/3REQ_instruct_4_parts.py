#!/usr/bin/env python
# coding: utf-8

# # 3REQ system
# 
# Requirement analysis system.
# 
# The flow is presented in the following figure: 
# 
# <img src="flow.png" alt="drawing" width="700"/>
# 
# Summary:
# 1. Extract sections that contain words like "latency"
# 2. Find if they resemble requirements or not
# 3. Check if they are specific kinds of requirements (like signalling)
# 4. Write new requirements based on the text in these sections. 

# In[ ]:

print("Welcome! ")


# This is only for Miroslaw
import os
os.environ['HF_HOME'] = '/mnt/i/cache'


# In[ ]:


# Importing required libraries

# docx2python is used to extract text, images, tables, and other data from .docx files
from docx2python import docx2python

# os module provides functions for interacting with the operating system
import os

# numpy is used for mathematical operations on large, multi-dimensional arrays and matrices
import numpy as np

# pandas is used for data manipulation and analysis
import pandas as pd

# TSNE from sklearn.manifold is used for dimensionality reduction
from sklearn.manifold import TSNE

# matplotlib.pyplot is used for creating static, animated, and interactive visualizations in Python
import matplotlib.pyplot as plt

# SentenceTransformer is used for training and using transformer models for generating sentence embeddings
from sentence_transformers import SentenceTransformer

# tqdm is used to make loops show a smart progress meter
from tqdm import tqdm

# torch is the main package in PyTorch, it provides a multi-dimensional array with support for autograd operations like backward()
import torch

# AutoModelForCausalLM, AutoTokenizer, pipeline are from the transformers library by Hugging Face which provides state-of-the-art machine learning models like BERT, GPT-2, etc.
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# euclidean distance and cosine distance
from scipy.spatial import distance

# random generator for the last figure
import random


# In[ ]:


print(torch.cuda.is_available())


# In[ ]:


# suppress warnings
import warnings

warnings.filterwarnings("ignore")


# ## Part 1: List the documents's sections with "latency"
# 
# In the first step, we go through the documents in the folder "input_standards" and we extract which sections of these documents contain th word "latency". We store the results in a dictionary.

# In[ ]:


def extractLatencySections(doc):
    strSectionTitle = ""
    dictSections = {}
    listLatency = []
    skippedSections = 0

    doc_result = docx2python(doc,paragraph_styles = True, html=True)

    # we iterate over all lines
    # look for the section titles (which have the tag <h1>, <h2>, <h3>, etc.)
    # then we add the content of each section to the dictionary
    # and if there is a word "latency" somewhere in the section, we add the section title to the listLatency
    for oneLine in tqdm(doc_result.text.split('\n')):
        if "<h" in oneLine:
            strSectionTitle = oneLine
            dictSections[strSectionTitle] = []

        if strSectionTitle != "":  
            dictSections[strSectionTitle].append(oneLine)

        keywordsInLine = ["latency"]
        keywordsInSections = ["references", "introduction", "definition", "abstract", "conclusion", "acknowledgements", "appendix", "table of contents", "table of figures", "table of tables", "table of contents", "table of figures", "table of tables", "bibliography", "index", "glossary", "list of figures", "list of tables", "list of abbreviations", "list of symbols", "list of terms", "list of equations", "list of algorithms", "list of acronyms", "list of illustrations", "list of appendices"]

        if any(word in oneLine.lower() for word in keywordsInLine) and not any(word in strSectionTitle.lower() for word in keywordsInSections): 
            listLatency.append(strSectionTitle)
        else:
            skippedSections += 1
            
            
    # remove the keys from the dictionary if they are not part of the listLatency
    # as we want to get only the relevant sections, i.e., the one with the word latency
    for key in list(dictSections.keys()):
        if key not in listLatency:
            del dictSections[key]

    # print("Skipped sections: ", skippedSections)

    # return the dictionary with the relevant sections
    return dictSections


# In[ ]:


docInputFolder = "./22_standards"

# this is the return list of all the lines in the document
lstAllLines = []

# for each .docx file in the input folder
# extract the sections with latency using the extractLatencySections function
# and print the sections
for doc in tqdm(os.listdir(docInputFolder)):    

    if doc.endswith(".docx"):
        #print(f"Processing {doc}")

        # since things can go wrong with the latency library, 
        # we use a try except block to avoid the program to stop
        try: 
            dictSections = extractLatencySections(os.path.join(docInputFolder, doc))
        
            # we list the content
            # as a long list of sections 
            for key in dictSections:

                lstOneLine = [key, doc]

                for line in dictSections[key]:
                    lstOneLine.append(line)
                    
                lstAllLines.append(lstOneLine)

        except Exception as e:
            print(f"Error with {doc}: {e}")


# In[ ]:


# choose the right model
# in order of size -- the last one is only for A6000 :) 

# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("sentence-t5-large")
# model = SentenceTransformer("sentence-transformers/gtr-t5-xxl")

# list with all embeddings for the sections
lstEmbeddings = []
iCounter = 0

for oneLine in tqdm(lstAllLines):

    # the content of the section starts on the third position of the list
    sentences = oneLine[3:]

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)
    
    # Print the average embeddings for all the sentences 
    # in this section
    avg_embeddings = embeddings.mean(axis=0)
    
    lstOneLine = [oneLine[0], oneLine[1], 2, str(sentences).replace("$", "_").replace("\n", "_"), avg_embeddings]

    lstEmbeddings.append(lstOneLine)

    iCounter += 1
    # the code below is for debug only
    # in case we want to stop the loop after a certain number of iterations
    #if iCounter == 100:
    #    break


# In[ ]:


# save the lstEmbeddings to an xlsx file topic_relevant.xlsx
df = pd.DataFrame(lstEmbeddings, columns=["Section", "Document", "Category", "Sentences", "Embeddings"])
df.to_excel("topic_relevant.xlsx", index=False)


# In[ ]:


# let's plot these average embeddings using t-SNE

# we create a list with the embeddings
lstEmbeddingsNP = np.array([x[4] for x in lstEmbeddings])

# we use t-SNE to reduce the dimensionality of the embeddings
tsne = TSNE(n_components=2, verbose=0, perplexity=12, n_iter=300)
tsne_results = tsne.fit_transform(lstEmbeddingsNP)

# we plot the t-SNE results
plt.figure(figsize=(16,10))

plt.scatter(tsne_results[:,0], tsne_results[:,1])

# Add labels to each dot
for i, label in enumerate([x[0] for x in lstEmbeddings]):
    plt.text(tsne_results[i, 0], tsne_results[i, 1], label[3:10])


# plt.show()


# ## Part 2: Requirements classes
# 
# Checking whether these are signalling, payload, etc. requirements 

# In[ ]:


# read the requirements from the excel file requirements.xlsx, worksheet LR
df = pd.read_excel("req_classes.xlsx", sheet_name="LR")

# convert to list
lstRequirements = df.values.tolist()
lstRequirements[0]

# now we calculate the embeddings for each of these requirements
lstEmbeddingsReq = []

for oneLine in tqdm(lstRequirements):
    
        # the content of the section starts on the third position of the list
        sentences = oneLine[1]
    
        # Sentences are encoded by calling model.encode()
        embeddings = model.encode(sentences)
        
        # Print the average embeddings for all the sentences 
        # in this section
        avg_embedding = embeddings
        
        lstOneLine = [oneLine[0], 'latency', oneLine[1], oneLine[1], avg_embedding]
    
        lstEmbeddingsReq.append(lstOneLine)


# In[ ]:


# now we calculate the euclidean distance between the requirements and the sections
# that are relevant
lstDist = []
lstRelevantDist = []

for oneLine in tqdm(lstEmbeddings):
    for oneLineReq in lstEmbeddingsReq:
        # euclidean distance between the two embeddings
        dist = distance.cosine(oneLine[4], oneLineReq[4])
        lstDist.append([oneLine[0], oneLine[1], oneLineReq[0], dist, oneLine[3]])

# now we sort the list by the distance
lstDist.sort(key=lambda x: x[2])

# and we print them
for i in range(len(lstDist)):
    # print(f"Section {lstDist[i][0]} is close to requirement {lstDist[i][2]} with distance {lstDist[i][3]:.2f}")
    # add this to a list
    lstRelevantDist.append([lstDist[i][0], lstDist[i][1], lstDist[i][2], lstDist[i][3], lstDist[i][4]])


# In[ ]:


# save the list to an Excel file
dfOutput = pd.DataFrame(lstRelevantDist, columns=["Section", "Document", "Requirement", "Distance", "Content"])

# sort it by section and document
dfOutput = dfOutput.sort_values(by=["Section", "Document"])

# average the distance in dfOutput per section, document and requirement
dfOutput["Distance"] = dfOutput["Distance"].astype(float)
dfGrouped = dfOutput.groupby(["Section", "Document", "Requirement", "Content"])

#convert dfGrouped to a dataframe
dfGrouped = dfGrouped.agg({"Distance": "mean"}).reset_index()


# In[ ]:


# now check the minimum distance per section and document
dfGrouped = dfGrouped.sort_values(by=["Section", "Document", "Distance"])

dfGrouped = dfGrouped.groupby(["Section", "Document"]).first().reset_index()


# In[ ]:


# now we turn this into a list of lists
lstRelevantDistGroup = dfGrouped.values.tolist()

print(len(lstRelevantDistGroup))


# In[ ]:


# convert to dataframe and save to excel lstRelevantDist
dfRelevantDist = pd.DataFrame(lstRelevantDistGroup, columns=["Section", "Document", "Requirement", "Content", "Distance"])

dfRelevantDist.to_excel("./classified_requirements.xlsx", index=False)


# ## Part 3: Check if the requirements exist in the database
# 
# In this step, we check if the text that we identified so far is covered by the requirements that exist in the database. We use the sentence transformers to get the embeddings of the text and then we compare them to the existin sections. 

# In[ ]:


thresholdPartial = 0.15
thresholdCovered = 0.20


# In[ ]:


# read the requirements from the Excel file
dfRelevantSections = pd.read_excel("./classified_requirements.xlsx")

lstRelevantSections = dfRelevantSections.values.tolist()


# In[ ]:


lstRelevantSections[0]


# In[ ]:


# model = SentenceTransformer("sentence-t5-large")
# model = SentenceTransformer("sentence-transformers/gtr-t5-xxl")

# list with all embeddings for the sections
lstRelevantEmbeddings = []
iCounter = 0

for oneLine in tqdm(lstRelevantSections):

    # the content of the section starts on the third position of the list
    sentences = oneLine[3]

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)
    
    # Print the average embeddings for all the sentences 
    # in this section
    # avg_embedding = np.mean(embeddings, axis=0)
    
    lstOneLine = [oneLine[0], oneLine[1], 2, str(sentences).replace("$", "_").replace("\n", "_"), embeddings]

    lstRelevantEmbeddings.append(lstOneLine)

    iCounter += 1


# In[ ]:


# now, read the requirements from the requirements database, file 20_requirements.xlsx
dfTRequirements = pd.read_excel("existing_requirements.xlsx")

# convert to list
lstTRequirements = dfTRequirements.values.tolist()


# In[ ]:


lstTRequirements[0]


# In[ ]:


# make the embeddings
# model = SentenceTransformer("sentence-t5-large")

# list with all embeddings for the sections
lstTRequirementsEmbeddings = []
iCounter = 0

for oneLine in tqdm(lstTRequirements):

    # the content of the section starts on the requirement text
    sentences = oneLine[1]

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)
    
    # Print the average embeddings for all the sentences 
    # in this section
    #avg_embedding = np.mean(embeddings, axis=0)
    
    lstOneLine = [oneLine[0], oneLine[1], 2, str(sentences).replace("$", "_").replace("\n", "_"), embeddings]

    lstTRequirementsEmbeddings.append(lstOneLine)

    iCounter += 1


# In[ ]:


# average the embeddings for all the lstTRequirementsEmbeddings
lstTRequirementsEmbeddingsNP = np.array([x[4] for x in lstTRequirementsEmbeddings])

tRequirementsAvgEmbeddings = np.mean(lstTRequirementsEmbeddingsNP, axis=0)


# In[ ]:


# now, calculate the average distance of all the relevant sections to the average requirements
lstDist = []

for oneLine in tqdm(lstRelevantEmbeddings):
    # euclidean distance between the two embeddings
    dist = distance.cosine(oneLine[4], tRequirementsAvgEmbeddings)
    lstDist.append([oneLine[0], oneLine[1], dist, oneLine[3]])

# now we sort the list by the distance
lstDist.sort(key=lambda x: x[3])
    


# In[ ]:


# create a bar plot for the distances lstDist[2]
plt.figure(figsize=(16,10))

# Create a bar plot
plt.bar([x[0] for x in lstDist], [x[2] for x in lstDist])

# Add labels to each dot
for i, label in enumerate([x[1] for x in lstDist]):
    plt.text(i, lstDist[i][2], label)

# add horizontal lines for the thresholds
plt.axhline(y=thresholdPartial, color='orange', linestyle='-')
plt.axhline(y=thresholdCovered, color='red', linestyle='-')

#plt.show()


# In[ ]:


# Visualize the results using the scatter plot, where
# the x-axis is the distance to the average requirement
# the y-axis is always 1
# the color is red for the requirements

# Create a color map based on x[2]
color_map = {0: 'blue', 1: 'green', 2: 'red'}
colors = [color_map[0] for x in lstDist]

# we plot the t-SNE results
plt.figure(figsize=(16,10))

random.seed(42)

plt.scatter([x[2] for x in lstDist], [random.random() * 2 for x in lstDist], c=colors, s=100, alpha=0.5)

# Add labels to each dot
#for i, label in enumerate([x[3] for x in lstDist]):
#    plt.text([x[2] for x in lstDist][i], 1, label[:20]+'...', rotation='vertical')

# Add vertical lines at 0.33 and 1.0
plt.axvline(x=thresholdPartial, color='orange', linestyle='--')
plt.axvline(x=thresholdCovered, color='red', linestyle='--')

# Set the limits of the x-axis
plt.xlim(0, 2)
plt.ylim(0, 2)

#plt.show()


# In[ ]:


# now we separate them into three lists:
# - with distance below 0.33 -- these requirements are covered
# - with distance between 0.33 and 1.0 -- these requirements are partially covered
# - with distance above 1.0 -- these requirements are not covered

lstCovered = []
lstPartiallyCovered = []
lstNotCovered = []

for oneLine in lstDist:
    if oneLine[2] < thresholdPartial:
        lstCovered.append(oneLine)
    elif oneLine[2] < thresholdCovered:
        lstPartiallyCovered.append(oneLine)
    else:
        lstNotCovered.append(oneLine)


# ## Part 4: create requirements based on each of these sections
# 
# In the last step, we create new requirements based on the sections identified in the previous steps.

# In[ ]:


torch.random.manual_seed(0)

modelInstr = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
    attn_implementation='eager',
)
tokenizerInstr = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")


# In[ ]:


def createRequirement(content, type, model, tokenizer):
    content1 = content.split(",")
    content1 = [x for x in content1[1:] if x not in ['', " ''", " '']"]]
    content_str = " ".join(content1)

    # this is about signalling, payload, c/c. 
    # typeStr = type.split("_")[1]
    
    # strContent = f"Based on this : {content_str}. Write the requirement in the following format 'The {typeStr} of the system shall ' "
    strContent = f"Based on this : {content_str}. Write the requirement in the following format 'The system shall ' "
    
    messages = [
        {"role": "user", "content": strContent},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    
    return output[0]['generated_text']


# In[ ]:


lstGenerated = []
iCounter = 0

# we generate new requirements for the partially covered ones
if len(lstPartiallyCovered) > 0:
    for oneLine in tqdm(lstPartiallyCovered):
        if (len(oneLine[3]) < 4095):
            strRequirement = createRequirement(oneLine[3], oneLine[2], modelInstr, tokenizerInstr)
            lstGenerated.append([oneLine[0], oneLine[1], oneLine[2], oneLine[3], strRequirement])


# In[ ]:


dfOutput = pd.DataFrame(lstGenerated, columns=["Section", "Document",  "Distance", "Content", "Generated requirement"])
dfOutput.to_excel("./output_partially_covered_generated.xlsx", index=False)


# In[ ]:


lstGenerated = []
iCounter = 0

# we generate new requirements for the not covered requirements
if len(lstNotCovered) > 0:
    for oneLine in tqdm(lstNotCovered):
        iCounter += 1
        if (len(oneLine[3]) < 10000):
            strRequirement = createRequirement(oneLine[3], oneLine[2], modelInstr, tokenizerInstr)
            lstGenerated.append([oneLine[0], oneLine[1], oneLine[2], oneLine[3], strRequirement])


# In[ ]:


dfOutput = pd.DataFrame(lstGenerated, columns=["Section", "Document",  "Distance", "Content", "Generated requirements"])
dfOutput.to_excel("./output_not_covered_generated.xlsx", index=False)


# In[ ]:


lstGenerated = []
iCounter = 0

# we generate new requirements for the not covered requirements
if len(lstCovered) > 0:
    for oneLine in tqdm(lstCovered):
        iCounter += 1
        if (len(oneLine[3]) < 10000):
            strRequirement = createRequirement(oneLine[3], oneLine[2], modelInstr, tokenizerInstr)
            lstGenerated.append([oneLine[0], oneLine[1], oneLine[2], oneLine[3], strRequirement])


# In[ ]:


dfOutput = pd.DataFrame(lstGenerated, columns=["Section", "Document",  "Distance", "Content", "Generated requirements"])
dfOutput.to_excel("./output_covered_generated.xlsx", index=False)

