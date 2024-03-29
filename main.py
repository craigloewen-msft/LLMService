# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import json

import torch
from utils import (load_tokenizer, load_model, load_peft_model, get_device, 
                   generate_text, run_prompt, check_adapter_path, generate_string)

def loadIssueData():
    # Load issue_test.json file
    with open('issue_data.json') as f:
        data = json.load(f)
    return data

def getInputStringList(inputIssueList):
    inputStringList = []
    for issue in inputIssueList:
        inputStringList.append(issue['issueDescription'])
    
    return inputStringList

def main(model_name, adapters_name, torch_dtype, quant_type):
    """
    The main execution function that loads the model, tokenizer, and runs the prompt.
    Args:
    model_name (str): The name of the model to load.
    adapters_name (str): Path to the adapters file.
    torch_dtype (torch.dtype): The data type for model weights (e.g., torch.bfloat16).
    quant_type (str): The quantization type to use.
    """
    check_adapter_path(adapters_name)
    tokenizer = load_tokenizer(model_name)

    model = load_model(model_name, torch_dtype, quant_type)
    model.resize_token_embeddings(len(tokenizer))
    
    model = load_peft_model(model, adapters_name)
    device = get_device()
    model.to(device)
    print(f"Model {model_name} loaded successfully on {device}")
    template = "Your job is to label GitHub issues to help teams identify which part of the product the issue is in. You should apply labels that best describe the area of the product that the issue is related to. Output them as a comma separated list, and do not output anything else besides a label list. If no labels apply output <None>\n\n===Issue Info===\n{}\n\n===Labels to apply===\n"

    resultString = generate_string(model, tokenizer, device, "Dogs!", template)
    print("RESULT STRING")
    print(resultString) 

if __name__ == "__main__":
    model_name = "../model-cache/mistralai/Mistral-7B"
    adapters_name = "../models/qlora/qlora/gpu-cpu_model/adapter"  # Ensure this path is correctly set before running
    torch_dtype = torch.bfloat16  # Set the appropriate torch data type
    quant_type = 'nf4'  # Set the appropriate quantization type

    try:
        main(model_name, adapters_name, torch_dtype, quant_type)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
