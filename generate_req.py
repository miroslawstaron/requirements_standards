import pandas as pd
import json
import requests

def ask_llm(paragraph):
    url = 'http://localhost:11435/api/generate'
    data = {
        "model": "llama3.1:70b",
        "prompt": prompts['generate_requirement'] + paragraph,
        "stream": False
        }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    json_data = json.loads(response.text)
    return json_data['response']

def generate_req(file):
    # Read the Paragraph column
    df = pd.read_csv(file , sep=';')
    column = df['Paragraph']
    # ask llm for each row in the column
    print('Generating requirements...')
    for i in range(len(column)):
        paragraph = column[i]
        response = ask_llm(paragraph)
        df.at[i, 'Requirement'] = response
    # Save the new dataframe to a new csv file
    df.to_csv('outputs/new_requirements.csv', sep=';', index=False)


with open('prompts.json', 'r') as f:
    prompts = json.load(f)

generate_req('outputs/latency_paragraphs.csv')






    