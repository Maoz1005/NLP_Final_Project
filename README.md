# Can LLMs Pass a "Reverse Turing Test"?

This repository contains the code used to reproduce all experiments from our work on AI-versus-human text classification.

## Repository Structure

- **datasets/**  
  Contains scripts for preparing and assembling the datasets used in the experiments.  
  - **MAGE dataset:** Aggregates examples from multiple benchmark datasets, including human-written texts from seven writing tasks (e.g., stories, news, scientific writing) and AI-generated texts from 27 LLMs such as ChatGPT, LLaMA, and BLOOM. Links to the original MAGE files are provided in the paper.  
  - **Kaggle Mix dataset:** Assembled from five publicly available Kaggle datasets, containing both AI-generated texts (from GPT-4, LLaMA-70B, Claude, Google PaLM, etc.) and human-written texts across multiple genres (essays, academic articles, news reports, etc.). Links to all source Kaggle datasets appear in the Appendix of the paper.

- **base_experiments/**  
  Scripts for running the baseline experiments (zero-shot and fine-tuned without prompts). Each model has its own script, specifying the configurations required for training and evaluation pipelines.

- **advanced_experiments/**  
  Scripts for running advanced experiments, including prompt-based zero-shot tests, efficiency analyses (minimal training examples and epochs), and robustness tests on challenging examples. Model-specific configurations are referenced from the baseline experiments.

## Environment and Dependencies

We used the following library versions for reproducibility:

- HuggingFace `Transformers` 4.56  
- HuggingFace `Datasets` 2.18  
- PyTorch 2.2.2+cu118  
- scikit-learn 1.7.1  
- pandas 2.3.2  
- numpy 1.26.4  

Ensure these versions are installed to replicate our experiments successfully.

## Notes

- All model configurations, training parameters, and evaluation procedures are in the scripts.  
- For MAGE and Kaggle Mix datasets, the relevant links and documentation are available in the paper (see main text and Appendix).  
