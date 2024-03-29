# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import json

import torch
from utils import (load_tokenizer, load_model, load_peft_model, get_device, 
                   generate_text, run_prompt, check_adapter_path, generate_string)
from flask import Flask, request, jsonify

model_name = "model-cache/mistralai/Mistral-7B"
adapters_name = "models/qlora/qlora/gpu-cpu_model/adapter"  # Ensure this path is correctly set before running
torch_dtype = torch.bfloat16  # Set the appropriate torch data type
quant_type = 'nf4'  # Set the appropriate quantization type

try:
    check_adapter_path(adapters_name)
    tokenizer = load_tokenizer(model_name)

    model = load_model(model_name, torch_dtype, quant_type)
    model.resize_token_embeddings(len(tokenizer))
    
    model = load_peft_model(model, adapters_name)
    device = get_device()
    model.to(device)
    print(f"Model {model_name} loaded successfully on {device}")
    template = "Your job is to label GitHub issues to help teams identify which part of the product the issue is in. You should apply labels that best describe the area of the product that the issue is related to. Output them as a comma separated list, and do not output anything else besides a label list. If no labels apply output <None>\n\n===Issue Info===\n{}\n\n===Labels to apply===\n"
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)

app = Flask(__name__)

@app.route('/getlabels', methods=['POST'])
def getlabels():
    try: 
        data = request.get_json()  # get the JSON data
        title = data.get('title')  # get the 'title' value
        body = data.get('body')  # get the 'body' value
        
        descriptionString = "# " + title + "\n\n" + body
        
        # Get LLM string
        resultString = generate_string(model, tokenizer, device, descriptionString, template)
        # Split result string by ,
        resultArray = resultString.split(",")
        # Return the label list
        return jsonify(resultArray)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/', methods=['GET'])
def home():
    return 'Hello World', 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
