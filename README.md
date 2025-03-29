# RGAR
Code for Paper RGAR: Recurrence Generation-augmented Retrieval for Factual-aware Medical Question Answering submitted to ACL 2025.

## Table of Contents

- [Introduction](#introduction)
- [Models](#models)
  - [Llama Models](#llama-models)
  - [Qwen Models](#qwen-models)
- [Datasets](#datasets)
  - [MIRAGE](#mirage)
  - [EHRNoteQA](#ehrnoteqa)
- [Retriever](#retriever)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Provide an overview of your project, its objectives, and any relevant background information.

## Models

### Llama Models

To access Llama models:

1. **Register an account on Hugging Face**: Visit [https://huggingface.co/](https://huggingface.co/) and create an account.
2. **Request access from Meta AI**: Navigate to the Llama model page, such as [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), and request access. Approval may take one or two business days.
3. **Create an access token**: Go to your Hugging Face [settings page](https://huggingface.co/settings/tokens) and generate a new access token.
4. **Authenticate via the terminal**: Run `huggingface-cli login` in your terminal and enter your access token.

For more details, refer to the [Llama-2-7b-chat-hf model card](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

### Qwen Models

Accessing Qwen models is straightforward and does not require special permissions:

1. **Find Qwen models on Hugging Face**: Search for "Qwen" in the [Hugging Face models section](https://huggingface.co/models?q=Qwen).
2. **Use the models**: You can directly use these models without any additional approval process.

## Datasets

### MIRAGE

The [MIRAGE benchmark](https://github.com/Teddy-XiongGZ/MIRAGE) offers a comprehensive dataset for evaluating Retrieval-Augmented Generation (RAG) systems in medical question answering. It utilizes the MedRAG toolkit to assess various RAG components. The benchmark data is available in the `benchmark.json` file within the repository.

For more information, visit the [MIRAGE GitHub repository](https://github.com/Teddy-XiongGZ/MIRAGE).

### EHRNoteQA

[EHRNoteQA](https://github.com/ji-youn-kim/EHRNoteQA) is a benchmark designed to evaluate Large Language Models (LLMs) using real-world clinical discharge summaries. To access this dataset:

1. **Review the dataset details**: Visit the [EHRNoteQA GitHub repository](https://github.com/ji-youn-kim/EHRNoteQA) for comprehensive information.
2. **Apply for access via PhysioNet**: Submit an application through [PhysioNet](https://physionet.org) to gain access to the dataset.

Ensure compliance with all data usage agreements and ethical guidelines when handling this dataset.

## Retriever

This project employs the MedCPT retriever for information retrieval tasks. MedCPT is tailored for medical contexts, ensuring accurate and relevant results.

## Installation

Please follow the requirements.txt to install necessary packages.

## Usage  

### Project Structure  
Before running the model, you should obtain the required models and corpus. The models should be placed in Hugging Face’s default directory, while the corpus should be organized as follows. Due to copyright restrictions, **EHRNoteQA** is not included in `benchmark.json`.  

```text
RGAR/
├── corpus/               
│   ├── textbooks/    # Collection of textbooks for retrieval tasks
│   ├── ...                  
├── log/              # Logs for tracking experiments and debugging
├── results/          # Output results and evaluation metrics
├── src/              # Source code implementation
├── benchmark.json    # Benchmark settings for evaluation
├── requirements.txt  # Required dependencies
├── README.md         # Project documentation
├── demo.ipynb        # Example usage and function demonstration
└── ...       
```

### Running the Model  
Example shell scripts for running the model are provided in the `shell/` directory. Before execution, move the required script to the root directory. Detailed function usage is demonstrated in `demo.ipynb`.  

## Tips  

To ensure reproducibility, all experiments are conducted with `temperature=0`. Additional key parameter settings include:  
- **Precision**: We use `torch.bfloat16` for efficient computation.  
- **Token Generation**: To mitigate repetitive generation issues in small LLMs, we set `max_new_tokens=4096` and `repetition_penalty=1.2`.  
- **Answer Extraction**: To support EHRNoteQA with five options (and potential future cases with more options), answers are extracted using regex-based methods implemented in the `extract_answer` function within `pipeline.py`.  

## Acknowledgment

This project is based on modifications of the original **MedRAG** code. We have explicitly marked the corresponding Python files where modifications were made. We sincerely appreciate the efforts of the open-source community and the existing work that made this project possible.



