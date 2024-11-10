# Evaluation Framework for Mental Health Chatbots

## Overview

This repository contains the implementation of the evaluation framework as described in our paper available at [arXiv](https://arxiv.org/abs/2408.04650). This framework aims to assess the safety and reliability of mental health chatbots using a combination of expert-driven benchmarks and automated evaluation methods. 

## Abstract

This study develops and validates an evaluation framework for mental health chatbots, focusing on their safety and reliability due to their increasing popularity and capability for context-aware support. The framework includes 100 benchmark questions with ideal responses, validated by mental health experts, and tested on a GPT-3.5-turbo-based chatbot. Evaluation methods explored include large language model-based scoring, an agentic approach, and embedding models. Our results underline the importance of expert validation and the effectiveness of real-time data access in improving chatbot reliability.

## File Descriptions

1. `answer.py` - Runs the benchmark questions through GPT-3.5-turbo and stores the results in a CSV file.
2. `evaluate.py` - Evaluates GPT-3.5-turbo answers using three LLM-based evaluation methods and stores results in separate CSV files.
3. `evaluate_similarity_vector.py` - Computes vector similarity for GPT-3.5-turbo answers.
4. `agent_method_code` - (To be added) Implements the agentic approach for real-time data evaluation.
5. `plot.py` - Generates plots as presented in the paper.
6. `prepare_results.py` - Extracts scores from LLM-based evaluation outputs and stores them in separated CSV files.
7. `check_normal.py` - Checks the normality of experts' scores against automated evaluation scores to determine appropriate statistical tests.
8. `ttest.py` - Runs paired t-tests between expert scores and other evaluation methods.
9. `Safety_Benchmark_Mental Health -Sheet1new.csv` - Contains the benchmark title, situation, question, and ideal response.
10. `Safety_Benchmark_Mental Health-ChatGPT 3.5.csv` - Stores GPT-3.5-turbo's answers to the benchmark questions.
11. `Results.xlsx` - Compiled final results for all evaluation methods.
12. `requirements.txt` - Contains requirements needed to run the codes.

## Setup and Running

### Requirements

- Python 3.x
- Libraries: the list of requirements can be found in `requirements.txt`

To install dependencies please run the following in your commandline:

```bash
pip install -r requirements.txt
```

## Citation

If you find this framework useful in your research, please consider citing our paper:

```bibtex
@article{park2024building,
  title={Building Trust in Mental Health Chatbots: Safety Metrics and LLM-Based Evaluation Tools},
  author={Park, Jung In and Abbasian, Mahyar and Azimi, Iman and Bounds, Dawn and Jun, Angela and Han, Jaesu and McCarron, Robert and Borelli, Jessica and Li, Jia and Mahmoudi, Mona and others},
  journal={arXiv preprint arXiv:2408.04650},
  year={2024}
}