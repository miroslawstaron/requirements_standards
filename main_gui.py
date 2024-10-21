import json
import os
import tkinter as tk
from tkinter import ttk
import filter_docs

# File path for the configuration file
CONFIG_FILE = "config.json"

# Function to load the configuration from the JSON file
def load_config():
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    return config


# Load configuration
config = load_config()

# Create the main window
root = tk.Tk()
root.title(config.get("title", "Generate Requirements"))

path_var=tk.StringVar(value=config['folder_path'])
output_var=tk.StringVar(value=config['latency_paragraphs'])
keyword_var=tk.StringVar(value=config['keywords'][0])
model_var=config['model_name']

def run():
    config = {
        'folder_path': path_var.get(),
        'latency_paragraphs': output_var.get(),
        'keywords': [keyword_var.get()],
        'ignored_sections': ["References", "Appendix", "Definitions", "Abbreviations"],
        'model_name': model_var,
        'verbose': False
    }
    #save config to config.json
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)
    filter_docs.execute_filtering(config)

def on_model_select(event):
    model_var = combo_box.get()

# Set window size
width = config.get("width", 500)
height = config.get("height", 350)
root.geometry(f"{width}x{height}")

path_label = tk.Label(root, text = 'Folder Path', font=('arial',10))
path_entry = tk.Entry(root,textvariable = path_var, font=('arial',10,'normal'), width=50)

output_label = tk.Label(root, text = 'Output file name', font = ('arial',10,'normal'))
output_entry=tk.Entry(root, textvariable = output_var, font = ('arial',10,'normal'), width=50)

keyword_label = tk.Label(root, text = 'Keyword', font = ('arial',10,'normal'))
keyword_entry=tk.Entry(root, textvariable = keyword_var, font = ('arial',10,'normal'), width=50)

model_label = tk.Label(root, text = 'Language Model', font = ('arial',10,'normal'))
model_entry=ttk.Combobox(root, values=["llama3.1", "llama3.1:70b", "llama3.1:405b"], width=50)
model_entry.set(model_var)

run_btn=tk.Button(root,text = 'Run', command = run, width=30)

path_label.grid(row=0,column=0, padx=5)
path_entry.grid(row=0,column=1, pady=10)
output_label.grid(row=1,column=0, padx=5)
output_entry.grid(row=1,column=1, pady=10)
keyword_label.grid(row=2,column=0, padx=5)
keyword_entry.grid(row=2,column=1, pady=10)
model_label.grid(row=3,column=0, padx=5)
model_entry.grid(row=3,column=1, pady=10)
run_btn.grid(row=4,column=1)

# Run the application
root.mainloop()