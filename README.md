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

will be updated before 3.10.
