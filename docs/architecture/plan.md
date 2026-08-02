Author: Zirui Ren

# Motivation

Architecture IQ benchmark 回答一个基础的问题：
LLM 是否已经具备对于最简单模型、最简单数据集的 Architecture Intuition 和 Learning Mechanics 理解能力？
因此，ArchitectureIQ V1 会故意选择最简单、完全可控、完全可合成（fully synthetic）的模型和数据集，而不是一开始就在复杂的大模型和真实语料上进行测试。
这样做有几个目的：
1. 消除真实数据集带来的复杂性，把问题聚焦到模型本身。
2. 所有实验都可以程序化生成，能够无限扩展数据。
3. 每个变量都完全可控，更容易分析模型真正理解了什么。
4. 为后续扩展到真实 LLM 和复杂训练任务建立基础 Benchmark。
ArchitectureIQ V1 本质上是一套 Architecture Intelligence IQ Test，而不是一个复杂的大模型 Benchmark。

# Benchmark Description

For each multiple choice question, we need to describe to the test taker the following:

- Dataset
  - A sample from the dataset pool
    - For now, the dataset pool has only one type of dataset (but can have many variants with different functions):
      - Regression to a univariate symbolic function on [0,1]. Training and test samples are random points on the interval.
      - The input and output are both 1-D scalars.
    - In the future, the dataset pool will be expanded.
  - Describe the dataset with:
    - Natural language.
    - The code used to synthesize the dataset in PyTorch. Make sure this code is what is really being used in the evaluation pipeline.
  - The objective to optimize (e.g. for regression, use regression MSE error).
  - Produce a `dataset_spec.json` containing information (dataset description & objective) for each question. Make sure all questions use the same format.

Given the target dataset, we can generate arbitrarily many candidate choices for solving that dataset:

- Candidate Choices: Each choice is independent. Each one has the following information:
  - Model: 
    - Sample from model pool (for now only use MLP)
      - MLP
        - With or without residual
        - With or without layer norm
          - Can differ across layers.
        - Depth
        - Width
          - Always use same width for all hidden layers, even without residual.
        - Different activation functions (ReLU, leaky ReLU with certain leaky rate, GeLU, SiLU)
          - Can differ across layers.
        - Initializations
          - For now only use default PyTorch init. To be expanded.
    - Describe the model (all the above information) with:
      - Natural language.
      - PyTorch code to implement the model as a torch.nn.Module. Make sure the code matches what is really running in the backend.
  - Optimizer
    - Sample from the optimizer pool:
      - SGD
      - Adam
      - AdamW
      - other typical optimizers
    - Optimizer hyperparamters (learning rate, weight decay, betas, etc), batch size.
    - Description with natural language and real running code.
  - Loss Function
    - Sample from the loss function pool (For each family in the dataset pool, the loss function pool might be different).
      - For the current 1-D regression task, the pool might be something like:
        - MSE
        - MSE with L1/L2 regularization
    - Description with natural language and real running code.
  - Produce a `candidate_spec.json` file, containing the candidate's information.

With the `dataset_spec.json` and `candidate_spec.json`, the full training setups for each candidate should be unambiguous.

Then for the ground truth of each choice, do the following:
- Really run each candidate setup for its ground truth
  - Save the result (the target objective metric, both final and progressive w.r.t. data points fed (can show curve)).
  - Filter: n_seed runs, average wins by threshold margin (significance)
  - Tag significance
- Rerun the candidate with multiple seeds to get the mean and std of the final metric.

Then we **carefully** sample from the choices (a pool of them), and combine them the multiple choice questions:
- Question Generation
  - The choices in one question should always use the same training sample budget (steps * batch_size).
    - Comparison is done with the same limited budget, not training until convergence.
    - This requires that, when generating samples, you can't use random budget. You have to consider what the multiple choice questions need.
    - In one question, the metric of the winning choice should differ **significantly** from the rest. You should carefully select from the choices (under the same training budget) with their mean and std error.
  - Provide two modes of question formulation:
    - 1. Given what kind of question we want to generate, generate new candidates satisfying that and sample significant comparisons from them.
    - 2. No new candidates generated. Sample from existing candidate results to form a new question.
  - We want both kinds of questions: where the choices differ in only architecture (use mode 1), and where many things can be different. Tag the different kinds of questions (likely use mode 2).

- Prompt generation
  - Combine all information into a single prompt to feed to the test taker (likely and LLM).
  - Ask for a single letter (e.g. A or B or C for 3-choice) as the response.

For any part of the question generation pipeline that uses natural language, prioritize using a string template. Report if you think an LLM API is necessary, so we can consider including it in the pipeline.

# File Structure

We want both kinds of results:
- Candidate pool
  - Store the candidates generated.
    - Candidates w.r.t. the same dataset should be stored under the same folder.
      - That folder should contain the `dataset_spec.json` of the corresponding dataset.
      - Within that folder, candidates with the same training budget should be stored under the same folder.
  - Each candidate is a folder, containing:
    - All its related code files. These should match what is in the spec and what is really run.
    - Its `candidate_spec.json`.
    - Its ground truth results (final metric & metric progressive results (likely a numpy array or a tensor stored)).

- Multiple choice questions
  - The target dataset (specify so that the pipeline knows which dataset path of candidates to look for).
  - The paths to the candidates (either newly-generated with the question or pre-existing).
  - The question prompt.