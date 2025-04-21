# Well-Organized Papers

<details>
<summary><a href="#1-test-time-scaling-method-parallel-sampling">1. Test-Time Scaling Method: Parallel Sampling</a></summary>

  - [1.1 Application I: (Mathematical) Reasoning](#11-application-i-mathematical-reasoning)
  - [1.2 Application II: Code](#12-application-ii-code)
  - [1.3 Application III: Multimodal](#13-application-iii-multimodal)
  - [1.4 Application IV: Safety](#14-application-iv-safety)
  - [1.5 Application V: RAG](#15-application-v-rag)
  - [1.6 Application VI: Evaluation](#16-application-vi-evaluation)

</details>

<details>
<summary><a href="#2-test-time-scaling-method-tree-search">2. Test-Time Scaling Method: Tree Search</a></summary>

  - [2.1 Application I: (Mathematical) Reasoning](#21-application-i-mathematical-reasoning)
  - [2.2 Application II: Code](#22-application-ii-code)
  - [2.3 Application III: Multimodal](#23-application-iii-multimodal)
  - [2.4 Application IV: Agent](#24-application-iv-agent)
  - [2.5 Application V: Safety](#25-application-v-safety)
  - [2.6 Application VI: RAG](#26-application-vi-rag)
  - [2.7 Application VII: Evaluation](#27-application-vii-evaluation)
  - [2.8 Special Topic: Process Reward Model 🔥](#28-special-topic-process-reward-model-)

</details>

<details>
<summary><a href="#3-test-time-scaling-method-multi-turn-correction">3. Test-Time Scaling Method: Multi-turn Correction</a></summary>


  - [3.1 Application I: (Mathematical) Reasoning](#31-application-i-mathematical-reasoning)
  - [3.2 Application II: Code](#32-application-ii-code)
  - [3.3 Application III: Multimodal](#33-application-iii-multimodal)
  - [3.4 Application IV: Agent](#34-application-iv-agent)
  - [3.5 Application V: Embodied AI](#35-application-v-embodied-ai)
  - [3.6 Application VI: Safety](#36-application-vi-safety)
  - [3.7 Application VII: Evaluation](#37-application-vii-evaluation)
  - [3.8 Special Topic: Critical Perspective 🔥](#38-special-topic-critical-perspective-)

</details>

<details>
<summary><a href="#4-test-time-scaling-method-long-chain-of-thought">4. Test-Time Scaling Method: (long) Chain-of-Thought</a></summary>

  - [4.1 Application I: (Mathematical) Reasoning](#41-application-i-mathematical-reasoning)
  - [4.2 Application II: Code](#42-application-ii-code)
  - [4.3 Application III: Multimodal](#43-application-iii-multimodal)
  - [4.4 Application IV: Agent](#44-application-iv-agent)
  - [4.5 Application V: Embodied AI](#45-application-v-embodied-ai)
  - [4.6 Application VI: Safety](#46-application-vi-safety)
  - [4.7 Application VII: RAG](#47-application-vii-rag)
  - [4.8 Application VIII: Evaluation](#48-application-viii-evaluation)
  - [4.9 Special Topic: Representational Complexity of Transformers with CoT 🔥](#49-special-topic-representational-complexity-of-transformers-with-cot-)

</details>

<details>
<summary><a href="#5-scaling-reinforcement-learning-for-long-cot-">5. Scaling Reinforcement Learning for Long CoT 🔥</a></summary>

  - [5.1 Application: Math & Code](#51-application-math--code)
  - [5.2 Application: Search](#52-application-search)
  - [5.3 Application III: Multimodal](#53-application-iii-multimodal)
  - [5.4 Papers Sorted by RL Components](#54-papers-sorted-by-rl-components)
    - [5.4.1 Training Algorithm](#541-training-algorithm)
    - [5.4.2 Reward Model](#542-reward-model)
    - [5.4.3 Base Model](#543-base-model)
    - [5.4.4 Training Data](#544-training-data)
    - [5.4.5 Multi-stage Training](#545-multi-stage-training)
    - [5.4.6 Evaluation](#546-evaluation)
    - [5.4.7 Analysis](#547-analysis)
  - [5.5 Infra](#55-infra)

</details>

<details>
<summary><a href="#6-supervised-learning-for-long-cot-">6. Supervised Learning for Long CoT 🔥</a></summary>

  - [6.1 long CoT Resource](#61-long-cot-resource)
  - [6.2 Analysis](#62-analysis)

</details>

<details>
<summary><a href="#7-self-improvement-with-test-time-scaling">7. Self-improvement with Test-Time Scaling</a></summary>

  - [7.1 Parallel Sampling](#71-parallel-sampling)
  - [7.2 Tree Search](#72-tree-search)
  - [7.3 Multi-turn Correction](#73-multi-turn-correction)
  - [7.4 Long CoT](#74-long-cot)

</details>

<details>
<summary><a href="#8-ensemble-of-test-time-scaling-method">8. Ensemble of Test-Time Scaling Method</a></summary>
</details>

<details>
<summary><a href="#9-inference-time-scaling-laws">9. Inference Time Scaling Laws</a></summary>
</details>

<details>
<summary><a href="#10-improving-scaling-efficiency">10. Improving Scaling Efficiency</a></summary>

  - [10.1 Parallel Sampling](#101-parallel-sampling)
  - [10.2 Tree Search](#102-tree-search)
  - [10.3 Multi-turn Correction](#103-multi-turn-correction)
  - [10.4 Long CoT](#104-long-cot)

</details>

<details>
<summary><a href="#11-latent-thoughts">11. Latent Thoughts</a></summary>
</details>

## 1. Test-Time Scaling Method: Parallel Sampling

### 1.1 Application I: (Mathematical) Reasoning

- Training Verifiers to Solve Math Word Problems [[Paper]](https://arxiv.org/abs/2110.14168) ![](https://img.shields.io/badge/arXiv-2021.10-red)
- Self-Consistency Improves Chain of Thought Reasoning in Language Models [[Paper]](https://openreview.net/pdf?id=1PL1NIMMrw) ![](https://img.shields.io/badge/ICLR-2023-blue)
- Let's Verify Step by Step [[Paper]](https://arxiv.org/abs/2305.20050) ![](https://img.shields.io/badge/ICLR-2024-blue)
- Improving Large Language Model Fine-tuning for Solving Math Problems [[Paper]](https://arxiv.org/abs/2310.10047) ![](https://img.shields.io/badge/arXiv-2023.10-red)
- Common 7B Language Models Already Possess Strong Math Capabilities [[Paper]](https://arxiv.org/abs/2403.04706) ![](https://img.shields.io/badge/arXiv-2024.03-red)
- Getting 50% (SoTA) on ARC-AGI with GPT-4o [[Paper]](https://gradientscience.org/arcagi/) ![](https://img.shields.io/badge/Blog-2025-red)



### 1.2 Application II: Code

- Natural Language to Code Translation with Execution [[Paper]](https://aclanthology.org/2022.emnlp-main.231) ![](https://img.shields.io/badge/EMNLP-2022-blue)
- Competition-Level Code Generation with AlphaCode [[Paper]](https://www.science.org/doi/10.1126/science.abq1158) ![](https://img.shields.io/badge/Science-2022-blue)
- CodeT: Code Generation with Generated Tests [[Paper]](https://openreview.net/pdf?id=ktrw68Cmu9c) ![](https://img.shields.io/badge/ICLR-2023-blue)
- AlphaCode 2 Technical Report [[Paper]](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf) ![](https://img.shields.io/badge/Report-2024-blue)
- Training Software Engineering Agents and Verifiers with SWE-Gym [[Paper]](https://arxiv.org/abs/2412.21139) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- S*: Test Time Scaling for Code Generation [[Paper]](https://arxiv.org/abs/2502.14382) ![](https://img.shields.io/badge/arXiv-2025.02-red)

### 1.3 Application III: Multimodal
- URSA: Understanding and Verifying Chain-of-thought Reasoning in Multimodal Mathematics [[Paper]](https://arxiv.org/abs/2501.04686) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step [[Paper]](https://arxiv.org/abs/2501.13926) ![](https://img.shields.io/badge/arXiv-2025.01-red)


### 1.4 Application IV: Safety

- SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models [[Paper]](https://aclanthology.org/2023.emnlp-main.557) ![](https://img.shields.io/badge/EMNLP-2023-blue)
- Leveraging Reasoning with Guidelines to Elicit and Utilize Knowledge for Enhancing Safety Alignment [[Paper]](https://arxiv.org/abs/2502.04040) ![](https://img.shields.io/badge/arXiv-2025.02-red)

### 1.5 Application V: RAG 
- Chain-of-Retrieval Augmented Generation [[Paper]](https://arxiv.org/abs/2501.14342) ![](https://img.shields.io/badge/arXiv-2025.01-red)


### 1.6 Application VI: Evaluation

- Crowd Comparative Reasoning: Unlocking Comprehensive Evaluations for LLM-as-a-Judge [[Paper]](https://arxiv.org/abs/2502.12501) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Inference-Time Scaling for Generalist Reward Modeling [[Paper]](https://arxiv.org/abs/2504.02495) ![](https://img.shields.io/badge/arXiv-2025.04-red)

## 2. Test-Time Scaling Method: Tree Search


### 2.1 Application I: (Mathematical) Reasoning
  


- Generative Language Modeling for Automated Theorem Proving [[Paper]](https://arxiv.org/abs/2009.03393) ![](https://img.shields.io/badge/arXiv-2020.09-red)  
- HyperTree Proof Search for Neural Theorem Proving [[Paper]](http://papers.nips.cc/paper_files/paper/2022/hash/a8901c5e85fb8e1823bbf0f755053672-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2022-blue)  
- Large Language Model Guided Tree-of-Thought [[Paper]](https://arxiv.org/abs/2305.08291) ![](https://img.shields.io/badge/arXiv-2023.05-red)  
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models [[Paper]](http://papers.nips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2023-blue)  
- Alphazero-like Tree-Search Can Guide Large Language Model Decoding and Training [[Paper]](https://arxiv.org/abs/2309.17179) ![](https://img.shields.io/badge/arXiv-2023.09-red)  
- Hypothesis Search: Inductive Reasoning with Language Models [[Paper]](https://arxiv.org/abs/2309.05660) ![](https://img.shields.io/badge/arXiv-2023.09-red)  
- Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models [[Paper]](https://arxiv.org/abs/2310.04406) ![](https://img.shields.io/badge/arXiv-2023.10-red)  
- ToolChain*: Efficient Action Space Navigation in Large Language Models with A* Search [[Paper]](https://arxiv.org/abs/2310.13227) ![](https://img.shields.io/badge/arXiv-2023.10-red)  
- Reasoning with Language Model is Planning with World Model [[Paper]](https://aclanthology.org/2023.emnlp-main.507) ![](https://img.shields.io/badge/EMNLP-2023-blue)  
- Self-Evaluation Guided Beam Search for Reasoning [[Paper]](http://papers.nips.cc/paper_files/paper/2023/hash/81fde95c4dc79188a69ce5b24d63010b-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2023-blue)  
- AlphaMath Almost Zero: Process Supervision without Process [[Paper]](https://arxiv.org/abs/2405.03553) ![](https://img.shields.io/badge/arXiv-2024.05-red)  
- Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning [[Paper]](https://arxiv.org/abs/2405.00451) ![](https://img.shields.io/badge/arXiv-2024.05-red)  
- MindStar: Enhancing Math Reasoning in Pre-Trained LLMs at Inference Time [[Paper]](https://arxiv.org/abs/2405.16265) ![](https://img.shields.io/badge/arXiv-2024.05-red)  
- Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing [[Paper]](https://arxiv.org/abs/2404.12253) ![](https://img.shields.io/badge/arXiv-2024.04-red)  
- Plan of Thoughts: Heuristic-Guided Problem Solving with Large Language Models [[Paper]](https://arxiv.org/abs/2404.19055) ![](https://img.shields.io/badge/arXiv-2024.04-red)  
- ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search [[Paper]](https://arxiv.org/abs/2406.03816) ![](https://img.shields.io/badge/arXiv-2024.06-red)  
- Step-Level Value Preference Optimization for Mathematical Reasoning [[Paper]](https://arxiv.org/abs/2406.10858) ![](https://img.shields.io/badge/arXiv-2024.06-red)  
- Accessing GPT-4 Level Mathematical Olympiad Solutions via Monte Carlo Tree Self-Refine with LLaMa-3 8B [[Paper]](https://arxiv.org/abs/2406.07394) ![](https://img.shields.io/badge/arXiv-2024.06-red)  
- Chain of Preference Optimization: Improving Chain-of-Thought Reasoning in LLMs [[Paper]](https://arxiv.org/abs/2406.09136) ![](https://img.shields.io/badge/arXiv-2024.06-red)  
- Q*: Improving Multi-Step Reasoning for LLMs with Deliberative Planning [[Paper]](https://arxiv.org/abs/2406.14283) ![](https://img.shields.io/badge/arXiv-2024.06-red)  
- DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search [[Paper]](https://arxiv.org/abs/2408.08152) ![](https://img.shields.io/badge/arXiv-2024.08-red)  
- Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers [[Paper]](https://arxiv.org/abs/2408.06195) ![](https://img.shields.io/badge/arXiv-2024.08-red)  
- Planning with MCTS: Enhancing Problem-Solving in Large Language Models [[Paper]](https://openreview.net/forum?id=sdpVfWOUQA) ![](https://img.shields.io/badge/OpenReview-2024-blue)  
- CPL: Critical Plan Step Learning Boosts LLM Generalization in Reasoning Tasks [[Paper]](https://arxiv.org/abs/2409.08642) ![](https://img.shields.io/badge/arXiv-2024.09-red)  
- Interpretable Contrastive Monte Carlo Tree Search Reasoning [[Paper]](https://arxiv.org/abs/2410.01707) ![](https://img.shields.io/badge/arXiv-2024.10-red)  
- rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking [[Paper]](https://arxiv.org/abs/2501.04519) ![](https://img.shields.io/badge/arXiv-2025.01-red)  
- BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving [[Paper]](https://arxiv.org/abs/2502.03438) ![](https://img.shields.io/badge/arXiv-2025.02-red)  




### 2.2 Application II: Code

- Planning with Large Language Models for Code Generation [[Paper]](https://openreview.net/pdf?id=Lr8cOOtYbfL) ![](https://img.shields.io/badge/ICLR-2023-blue)
- Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models [[Paper]](https://arxiv.org/abs/2310.04406) ![](https://img.shields.io/badge/arXiv-2023.10-red)
- Cruxeval: A benchmark for code reasoning, understanding and execution [[Paper]](https://arxiv.org/abs/2401.03065) ![](https://img.shields.io/badge/arXiv-2024.01-red)
- Planning In Natural Language Improves LLM Search For Code Generation [[Paper]](https://arxiv.org/abs/2409.03733) ![](https://img.shields.io/badge/arXiv-2024.09-red)
- RethinkMCTS: Refining Erroneous Thoughts in Monte Carlo Tree Search for Code Generation [[Paper]](https://arxiv.org/abs/2409.09584) ![](https://img.shields.io/badge/arXiv-2024.09-red)
- SRA-MCTS: Self-driven Reasoning Augmentation with Monte Carlo Tree Search for Code Generation [[Paper]](https://arxiv.org/abs/2411.11053) ![](https://img.shields.io/badge/arXiv-2024.11-red)
- O1-Coder: An O1 Replication for Coding [[Paper]](https://arxiv.org/abs/2412.00154) ![](https://img.shields.io/badge/arXiv-2024.12-red)



### 2.3 Application III: Multimodal
- Llava-CoT: Let vision language models reason step-by-step [[Paper]](https://arxiv.org/abs/2411.10440) ![](https://img.shields.io/badge/arXiv-2024.11-red)
- Scaling inference-time search with vision value model for improved visual comprehension [[Paper]](https://arxiv.org/abs/2412.03704) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search [[Paper]](https://arxiv.org/abs/2412.18319) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- Llamav-o1: Rethinking step-by-step visual reasoning in llms [[Paper]](https://arxiv.org/abs/2501.06186) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Video-T1: Test-Time Scaling for Video Generation [[Paper]](https://arxiv.org/abs/2503.18942) ![](https://img.shields.io/badge/arXiv-2025.03-red)



### 2.4 Application IV: Agent
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models [[Paper]](http://papers.nips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2023-blue)  
- Don't Generate, Discriminate: A Proposal for Grounding Language Models to Real-World Environments [[Paper]](https://aclanthology.org/2023.acl-long.270) ![](https://img.shields.io/badge/ACL-2023-blue)  
- Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models [[Paper]](https://arxiv.org/abs/2310.04406) ![](https://img.shields.io/badge/arXiv-2023.10-red)  
- ToolChain*: Efficient Action Space Navigation in Large Language Models with A* Search [[Paper]](https://arxiv.org/abs/2310.13227) ![](https://img.shields.io/badge/arXiv-2023.10-red)
- Tree Search for Language Model Agents [[Paper]](https://arxiv.org/abs/2407.01476) ![](https://img.shields.io/badge/arXiv-2024.07-red)
- Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents [[Paper]](https://arxiv.org/abs/2408.07199) ![](https://img.shields.io/badge/arXiv-2024.08-red)




### 2.5 Application V: Safety


- C-MCTS: Safe Planning with Monte Carlo Tree Search [[Paper]](https://arxiv.org/abs/2305.16209) ![](https://img.shields.io/badge/arXiv-2023.05-red)
- Don't Throw Away Your Value Model! Generating More Preferable Text with Value-Guided Monte-Carlo Tree Search Decoding [[Paper]](https://arxiv.org/abs/2309.15028) ![](https://img.shields.io/badge/arXiv-2023.09-red)
- ARGS: Alignment as Reward-Guided Search [[Paper]](https://arxiv.org/abs/2402.01694) ![](https://img.shields.io/badge/ICLR-2024-blue)
- Think More, Hallucinate Less: Mitigating Hallucinations via Dual Process of Fast and Slow Thinking [[Paper]](https://arxiv.org/abs/2501.01306) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Almost Surely Safe Alignment of Large Language Models at Inference-Time [[Paper]](https://arxiv.org/abs/2502.01208) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- STAIR: Improving Safety Alignment with Introspective Reasoning [[Paper]](https://arxiv.org/abs/2502.02384) ![](https://img.shields.io/badge/arXiv-2025.02-red)

### 2.6 Application VI: RAG

- AirRAG: Activating Intrinsic Reasoning for Retrieval Augmented Generation using Tree-based Search [[Paper]](https://arxiv.org/abs/2501.10053) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Chain-of-Retrieval Augmented Generation [[Paper]](https://arxiv.org/abs/2501.14342) ![](https://img.shields.io/badge/arXiv-2025.01-red)


### 2.7 Application VII: Evaluation

- MCTS-Judge: Test-Time Scaling in LLM-as-a-Judge for Code Correctness Evaluation [[Paper]](https://arxiv.org/abs/2502.12468) ![](https://img.shields.io/badge/arXiv-2025.02-red)



### 2.8 Special Topic: Process Reward Model 🔥

- Solving math word problems with process- and outcome-based feedback [[Paper]](https://arxiv.org/abs/2211.14275) ![](https://img.shields.io/badge/arXiv-2022.11-red)
- Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations [[Paper]](https://arxiv.org/abs/2312.08935) ![](https://img.shields.io/badge/arXiv-2023.12-red)
- GLoRe: When, Where, and How to Improve LLM Reasoning via Global and Local Refinements [[Paper]](https://arxiv.org/abs/2402.10963) ![](https://img.shields.io/badge/arXiv-2024.02-red)
- Multi-step Problem Solving Through a Verifier: An Empirical Analysis on Model-induced Process Supervision [[Paper]](https://arxiv.org/abs/2402.02658) ![](https://img.shields.io/badge/arXiv-2024.02-red)
- Evaluating Mathematical Reasoning Beyond Accuracy [[Paper]](https://arxiv.org/abs/2404.05692) ![](https://img.shields.io/badge/arXiv-2024.04-red)
- AutoPSV: Automated Process-Supervised Verifier [[Paper]](https://arxiv.org/abs/2405.16802) ![](https://img.shields.io/badge/arXiv-2024.05-red)
- Improve Mathematical Reasoning in Language Models by Automated Process Supervision [[Paper]](https://arxiv.org/abs/2406.06592) ![](https://img.shields.io/badge/arXiv-2024.06-red)
- Token-Supervised Value Models for Enhancing Mathematical Reasoning Capabilities of Large Language Models [[Paper]](https://arxiv.org/abs/2407.12863) ![](https://img.shields.io/badge/arXiv-2024.07-red)
- Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning [[Paper]](https://arxiv.org/abs/2410.08146) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- Free Process Rewards without Process Labels [[Paper]](https://arxiv.org/abs/2412.01981) ![](https://img.shields.io/badge/arXiv-2024.12-red)




## 3. Test-Time Scaling Method: Multi-turn Correction


### 3.1 Application I: (Mathematical) Reasoning


- Generating Sequences by Learning to Self-Correct [[Paper]](https://openreview.net/pdf?id=hH36JeQZDaO) ![](https://img.shields.io/badge/ICLR-2023-blue)
- Baldur: Whole-Proof Generation and Repair with Large Language Models [[Paper]](https://arxiv.org/abs/2303.04910) ![](https://img.shields.io/badge/arXiv-2023.03-red)
- CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing [[Paper]](https://arxiv.org/abs/2305.11738) ![](https://img.shields.io/badge/arXiv-2023.05-red)
- Improving Factuality and Reasoning in Language Models through Multiagent Debate [[Paper]](https://arxiv.org/abs/2305.14325) ![](https://img.shields.io/badge/arXiv-2023.05-red)
- Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate [[Paper]](https://arxiv.org/abs/2305.19118) ![](https://img.shields.io/badge/arXiv-2023.05-red)
- Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework [[Paper]](https://aclanthology.org/2023.acl-long.320) ![](https://img.shields.io/badge/ACL-2023-blue)
- Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-Based Self-Verification [[Paper]](https://arxiv.org/abs/2308.07921) ![](https://img.shields.io/badge/arXiv-2023.08-red)
- Language Models can Solve Computer Tasks [[Paper]](http://papers.nips.cc/paper_files/paper/2023/hash/7cc1005ec73cfbaac9fa21192b622507-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2023-blue)
- Self-Refine: Iterative Refinement with Self-Feedback [[Paper]](http://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2023-blue)
- Reflexion: language agents with verbal reinforcement learning [[Paper]](http://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2023-blue)
- REFINER: Reasoning Feedback on Intermediate Representations [[Paper]](https://aclanthology.org/2024.eacl-long.67) ![](https://img.shields.io/badge/EACL-2024-blue)
- Debating with More Persuasive LLMs Leads to More Truthful Answers [[Paper]](https://arxiv.org/abs/2402.06782) ![](https://img.shields.io/badge/arXiv-2024.02-red)
- Recursive Introspection: Teaching Language Model Agents How to Self-Improve [[Paper]](https://arxiv.org/abs/2407.18219) ![](https://img.shields.io/badge/arXiv-2024.07-red)
- Training Language Models to Self-Correct via Reinforcement Learning [[Paper]](https://arxiv.org/abs/2409.12917) ![](https://img.shields.io/badge/arXiv-2024.09-red)
- Enhancing LLM Reasoning via Critique Models with Test-Time and Training-Time Supervision [[Paper]](https://arxiv.org/abs/2411.16579) ![](https://img.shields.io/badge/arXiv-2024.11-red)


### 3.2 Application II: Code


- Teaching Large Language Models to Self-Debug [[Paper]](https://arxiv.org/abs/2304.05128) ![](https://img.shields.io/badge/arXiv-2023.04-red)
- Is Self-Repair a Silver Bullet for Code Generation? [[Paper]](https://arxiv.org/abs/2306.09896) ![](https://img.shields.io/badge/arXiv-2023.06-red)
- Reflexion: language agents with verbal reinforcement learning [[Paper]](http://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2023-blue)
- Phenomenal Yet Puzzling: Testing Inductive Reasoning Capabilities of Language Models with Hypothesis Refinement [[Paper]](https://arxiv.org/abs/2310.08559) ![](https://img.shields.io/badge/arXiv-2023.10-red)
- Self-taught optimizer (stop): Recursively self-improving code generation [[Paper]](https://openreview.net/forum?id=1gkePTsAWf) ![](https://img.shields.io/badge/COLM-2024-blue)
- LLM Critics Help Catch LLM Bugs [[Paper]](https://arxiv.org/abs/2407.00215) ![](https://img.shields.io/badge/arXiv-2024.07-red)


### 3.3 Application III: Multimodal

- Vision-Language Models Can Self-Improve Reasoning via Reflection [[Paper]](https://arxiv.org/abs/2411.00855) ![](https://img.shields.io/badge/arXiv-2024.11-red)
- Insight-v: Exploring long-chain visual reasoning with multimodal large language models [[Paper]](https://arxiv.org/abs/2411.14432) ![](https://img.shields.io/badge/arXiv-2024.11-red)
- Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step [[Paper]](https://arxiv.org/abs/2501.13926) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- MINT: Multi-modal Chain of Thought in Unified Generative Models for Enhanced Image Generation [[Paper]](https://arxiv.org/abs/2503.01298) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing [[Paper]](https://arxiv.org/abs/2503.10639) ![](https://img.shields.io/badge/arXiv-2025.03-red)


### 3.4 Application IV: Agent

- Language Models can Solve Computer Tasks [[Paper]](http://papers.nips.cc/paper_files/paper/2023/hash/7cc1005ec73cfbaac9fa21192b622507-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2023-blue)
- Reflexion: language agents with verbal reinforcement learning [[Paper]](http://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2023-blue)
- Autonomous Evaluation and Refinement of Digital Agents [[Paper]](https://arxiv.org/abs/2404.06474) ![](https://img.shields.io/badge/arXiv-2024.04-red)


### 3.5 Application V: Embodied AI

- Inner Monologue: Embodied Reasoning through Planning with Language Models [[Paper]](https://arxiv.org/abs/2207.05608) ![](https://img.shields.io/badge/arXiv-2022.07-red)
- REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction [[Paper]](https://aclanthology.org/2023.emnlp-main.557) ![](https://img.shields.io/badge/CoRL-2023-blue)
- Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners [[Paper]](https://proceedings.mlr.press/v164/) ![](https://img.shields.io/badge/CoRL-2023-blue)



### 3.6 Application VI: Safety
- Generating Sequences by Learning to Self-Correct [[Paper]](https://openreview.net/pdf?id=hH36JeQZDaO) ![](https://img.shields.io/badge/ICLR-2023-blue)
- CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing [[Paper]](https://arxiv.org/abs/2305.11738) ![](https://img.shields.io/badge/arXiv-2023.05-red)
- Improving Factuality and Reasoning in Language Models through Multiagent Debate [[Paper]](https://arxiv.org/abs/2305.14325) ![](https://img.shields.io/badge/arXiv-2023.05-red)
- MART: Improving LLM Safety with Multi-round Automatic Red-Teaming [[Paper]](https://aclanthology.org/2024.naacl-long.107) ![](https://img.shields.io/badge/NAACL-2024-blue)
- Combating Adversarial Attacks with Multi-Agent Debate [[Paper]](https://arxiv.org/abs/2401.05998) ![](https://img.shields.io/badge/arXiv-2024.01-red)
- Debategpt: Fine-tuning large language models with multi-agent debate supervision [[Paper]](https://openreview.net/forum?id=ChNy95ovpF) ![](https://img.shields.io/badge/OpenReview-2024-blue)

### 3.7 Application VII: Evaluation


- ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate [[Paper]](https://arxiv.org/abs/2308.07201) ![](https://img.shields.io/badge/arXiv-2023.08-red)
- Can Large Language Models be Trusted for Evaluation? Scalable Meta-Evaluation of LLMs as Evaluators via Agent Debate [[Paper]](https://arxiv.org/abs/2401.16788) ![](https://img.shields.io/badge/arXiv-2024.01-red)

### 3.8 Special Topic: Critical Perspective 🔥

- Large Language Models Cannot Self-Correct Reasoning Yet [[Paper]](https://arxiv.org/abs/2310.01798) ![](https://img.shields.io/badge/arXiv-2023.10-red)
- Can Large Language Models Really Improve by Self-Critiquing Their Own Plans? [[Paper]](https://arxiv.org/abs/2310.08118) ![](https://img.shields.io/badge/arXiv-2023.10-red)
- GPT-4 Doesn't Know It's Wrong: An Analysis of Iterative Prompting for Reasoning Problems [[Paper]](https://arxiv.org/abs/2310.12397) ![](https://img.shields.io/badge/arXiv-2023.10-red)
- LLMs cannot find reasoning errors, but can correct them given the error location [[Paper]](https://arxiv.org/abs/2311.08516) ![](https://img.shields.io/badge/arXiv-2023.11-red)
- Pride and Prejudice: LLM Amplifies Self-Bias in Self-Refinement [[Paper]](https://arxiv.org/abs/2402.11436) ![](https://img.shields.io/badge/arXiv-2024.02-red)
- When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs [[Paper]](https://aclanthology.org/2024.tacl-1.78/) ![](https://img.shields.io/badge/TACL-2024-blue)


## 4. Test-Time Scaling Method: (long) Chain-of-Thought


### 4.1 Application I: (Mathematical) Reasoning




- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models [[Paper]](http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2022-blue)
- OpenAI O1 System Card [[Blog]](https://openai.com/index/openai-o1-system-card/) ![](https://img.shields.io/badge/Blog-2024-red)
- QwQ: Reflect Deeply on the Boundaries of the Unknown [[Blog]](https://qwenlm.github.io/blog/qwq-32b-preview/) ![](https://img.shields.io/badge/Blog-2024-red)
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning [[Paper]](https://arxiv.org/abs/2501.12948) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Kimi k1.5: Scaling reinforcement learning with llms [[Paper]](https://arxiv.org/abs/2501.12599) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling [[Paper]](https://arxiv.org/abs/2501.11651) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- There May Not be Aha Moment in R1-Zero-like Training — A Pilot Study [[Blog]](https://oatllm.notion.site/oat-zero) ![](https://img.shields.io/badge/Notion-2025-red)
- 7B Model and 8K Examples: Emerging Reasoning with Reinforcement Learning is Both Effective and Efficient [[Blog]](https://hkust-nlp.notion.site/simplerl-reason) ![](https://img.shields.io/badge/Notion-2025-red)
- s1: Simple test-time scaling [[Paper]](https://arxiv.org/abs/2501.19393) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- LIMO: Less is More for Reasoning [[Paper]](https://arxiv.org/abs/2502.03387) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning [[Paper]](https://arxiv.org/abs/2502.14768) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Demystifying Long Chain-of-Thought Reasoning in LLMs [[Paper]](https://arxiv.org/abs/2502.03373) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution [[Paper]](https://arxiv.org/abs/2502.18449) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- LIMR: Less is More for RL Scaling [[Paper]](https://arxiv.org/abs/2502.11886) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale [[Paper]](https://arxiv.org/abs/2503.14476) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- An Empirical Study on Eliciting and Improving R1-like Reasoning Models [[Paper]](https://arxiv.org/abs/2503.04548) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Open-Reasoner-Zero: An Open Source Approach to Scaling Reinforcement Learning on the Base Model [[Github]](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) ![](https://img.shields.io/badge/GitHub-2025-blue)
- DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL [[Blog]](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) ![](https://img.shields.io/badge/Notion-2025-blue)
- Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs [[Paper]](https://arxiv.org/abs/2503.01307) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Understanding R1-Zero-Like Training: A Critical Perspective [[Paper]](https://arxiv.org/abs/2503.20783) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Can Better Cold-Start Strategies Improve RL Training for LLMs? [[Blog]](https://tangible-polo-203.notion.site/Can-Better-Cold-Start-Strategies-Improve-RL-Training-for-LLMs-17aa0742a51680828616c867ed53bc6b) ![](https://img.shields.io/badge/Notion-2025-blue)
- What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret [[Paper]](https://arxiv.org/abs/2503.01491) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't [[Paper]](https://arxiv.org/abs/2503.16219) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning [[Paper]](https://arxiv.org/abs/2504.02546) ![](https://img.shields.io/badge/arXiv-2025.04-red)
- VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks [[Paper]](https://arxiv.org/abs/2504.05118) ![](https://img.shields.io/badge/arXiv-2025.04-red)
- Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining [[Paper]](https://arxiv.org/abs/2504.07912) ![](https://img.shields.io/badge/arXiv-2025.04-red)
- Rethinking Reflection in Pre-Training [[Paper]](https://arxiv.org/abs/2504.04022) ![](https://img.shields.io/badge/arXiv-2025.04-red)






### 4.2 Application II: Code

- OpenAI O1 System Card [[Paper]](https://openai.com/index/openai-o1-system-card/) ![](https://img.shields.io/badge/Blog-2024-red)
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning [[Paper]](https://arxiv.org/abs/2501.12948) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Kimi k1.5: Scaling reinforcement learning with llms [[Paper]](https://arxiv.org/abs/2501.12599) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- s1: Simple test-time scaling [[Paper]](https://arxiv.org/abs/2501.19393) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Competitive Programming with Large Reasoning Models [[Paper]](https://arxiv.org/abs/2502.06807) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution [[Paper]](https://arxiv.org/abs/2502.18449) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Thinking Longer, Not Larger: Enhancing Software Engineering Agents via Scaling Test-Time Compute [[Paper]](https://arxiv.org/abs/2503.23803) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- ToRL: Scaling Tool-Integrated RL [[Paper]](https://arxiv.org/abs/2503.23383) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- OpenCodeReasoning: Advancing Data Distillation for Competitive Coding [[Paper]](https://arxiv.org/abs/2504.01943) ![](https://img.shields.io/badge/arXiv-2025.04-red)
- DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level [[Paper]](https://www.together.ai/blog/deepcoder) ![](https://img.shields.io/badge/Blog-2025-red)
- Seed-Thinking-v1.5: Advancing Superb Reasoning Models with Reinforcement Learning [[Paper]](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5) ![](https://img.shields.io/badge/GitHub-2025-red)




### 4.3 Application III: Multimodal




- QVQ: To See the World with Wisdom [[Paper]](https://qwenlm.github.io/blog/qvq-72b-preview/) ![](https://img.shields.io/badge/Blog-2024-red)
- Llava-onevision: Easy visual task transfer [[Paper]](https://arxiv.org/abs/2408.03326) ![](https://img.shields.io/badge/arXiv-2024.08-red)
- Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution [[Paper]](https://arxiv.org/abs/2409.12191) ![](https://img.shields.io/badge/arXiv-2024.09-red)
- Reducing hallucinations in vision-language models via latent space steering [[Paper]](https://arxiv.org/abs/2410.15778) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale [[Paper]](https://arxiv.org/abs/2412.05237) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- Virgo: A Preliminary Exploration on Reproducing o1-like MLLM [[Paper]](https://arxiv.org/abs/2501.01904) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Imagine while Reasoning in Space: Multimodal Visualization-of-Thought [[Paper]](https://arxiv.org/abs/2501.07542) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Kimi k1.5: Scaling reinforcement learning with llms [[Paper]](https://arxiv.org/abs/2501.12599) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Qwen2.5-VL Technical Report [[Paper]](https://arxiv.org/abs/2502.13923) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework [[Github]](https://github.com/hiyouga/EasyR1) ![](https://img.shields.io/badge/GitHub-2025-red)
- LMM-R1 [[Github]](https://github.com/TideDra/lmm-r1) ![](https://img.shields.io/badge/GitHub-2025-red)
- VLM-R1: A stable and generalizable R1-style Large Vision-Language Model [[Github]](https://github.com/om-ai-lab/VLM-R1) ![](https://img.shields.io/badge/GitHub-2025-red)
- R1-V: Reinforcing Super Generalization Ability in Vision-Language Models with Less Than \$3 [[Github]](https://github.com/Deep-Agent/R1-V) ![](https://img.shields.io/badge/GitHub-2025-red)
- open-r1-multimodal: A fork to add multimodal model training to open-r1 [[Github]](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) ![](https://img.shields.io/badge/GitHub-2025-red)
- R1-Multimodal-Journey: A journey to real multimodal R1 [[Github]](https://github.com/FanqingM/R1-Multimodal-Journey) ![](https://img.shields.io/badge/GitHub-2025-red)
- Open-R1-Video [[Github]](https://github.com/Wang-Xiaodong1899/Open-R1-Video) ![](https://img.shields.io/badge/GitHub-2025-red)
- R1-Onevision: Open-Source Multimodal Large Language Model with Reasoning Ability [[Notion]](https://yangyi-vai.notion.site/r1-onevision) ![](https://img.shields.io/badge/Notion-2025-red)
- Video-R1: Reinforcing Video Reasoning in MLLMs [[Paper]](https://arxiv.org/abs/2503.21776) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model [[Paper]](https://arxiv.org/abs/2503.05132) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- MM-Eureka: Exploring the Frontiers of Multimodal Reasoning with Rule-based Reinforcement Learning [[Paper]](https://arxiv.org/abs/2503.07365) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement [[Paper]](https://arxiv.org/abs/2503.06520) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Rethinking RL Scaling for Vision Language Models: A Transparent, From-Scratch Framework and Comprehensive Evaluation Scheme [[Paper]](https://arxiv.org/abs/2504.02587) ![](https://img.shields.io/badge/arXiv-2025.04-red)
- Improved Visual-Spatial Reasoning via R1-Zero-Like Training [[Paper]](https://arxiv.org/abs/2504.00883) ![](https://img.shields.io/badge/arXiv-2025.04-red)
- Kimi-VL [[Github]](https://github.com/MoonshotAI/Kimi-VL) ![](https://img.shields.io/badge/Github-2025-red)
- Introducing OpenAI o3 and o4-mini [[Blog]](https://openai.com/index/introducing-o3-and-o4-mini/) ![](https://img.shields.io/badge/Blog-2025-red)



### 4.4 Application IV: Agent


- ReAct: Synergizing Reasoning and Acting in Language Models [[Paper]](https://openreview.net/pdf?id=WE_vluYUL-X) ![](https://img.shields.io/badge/ICLR-2023-blue)
- Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku [[Blog]](https://www.anthropic.com/news/3-5-models-and-computer-use) ![](https://img.shields.io/badge/Blog-2024-red)
- PC Agent: While You Sleep, AI Works -- A Cognitive Journey into Digital World [[Paper]](https://arxiv.org/abs/2412.17589) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- UI-TARS: Pioneering Automated GUI Interaction with Native Agents [[Paper]](https://arxiv.org/abs/2501.12326) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Introducing deep research [[Blog]](https://openai.com/index/introducing-deep-research/) ![](https://img.shields.io/badge/Blog-2025-red)
- Computer-Using Agent [[Blog]](https://openai.com/index/computer-using-agent/) ![](https://img.shields.io/badge/Blog-2025-red)
- The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks [[Paper]](https://arxiv.org/abs/2502.08235) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution [[Paper]](https://arxiv.org/abs/2502.18449) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments [[Paper]](https://arxiv.org/abs/2504.03160) ![](https://img.shields.io/badge/arXiv-2025.04-red)



### 4.5 Application V: Embodied AI

- Agent Planning with World Knowledge Model [[Paper]](https://arxiv.org/abs/2405.14205) ![](https://img.shields.io/badge/arXiv-2024.05-red)
- Robotic control via embodied chain-of-thought reasoning [[Paper]](https://arxiv.org/abs/2407.08693) ![](https://img.shields.io/badge/arXiv-2024.07-red)
- Improving Vision-Language-Action Models via Chain-of-Affordance [[Paper]](https://arxiv.org/abs/2412.20451) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- SpatialCoT: Advancing Spatial Reasoning through Coordinate Alignment and Chain-of-Thought for Embodied Task Planning [[Paper]](https://arxiv.org/abs/2501.10074) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Action-Free Reasoning for Policy Generalization [[Paper]](https://arxiv.org/abs/2502.03729) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning [[Paper]](https://arxiv.org/abs/2503.15558) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Gemini Robotics: Bringing AI into the Physical World [[Paper]](https://arxiv.org/abs/2503.20020) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models [[Paper]](https://arxiv.org/abs/2503.22020) ![](https://img.shields.io/badge/arXiv-2025.03-red)

### 4.6 Application VI: Safety

- Chain-of-Verification Reduces Hallucination in Large Language Models [[Paper]](https://arxiv.org/abs/2309.11495) ![](https://img.shields.io/badge/arXiv-2023.09-red)
- Mixture of insighTful Experts (MoTE): The Synergy of Thought Chains and Expert Mixtures in Self-Alignment [[Paper]](https://arxiv.org/abs/2405.00557) ![](https://img.shields.io/badge/arXiv-2024.05-red)
- Deliberative Alignment: Reasoning Enables Safer Language Models [[Paper]](https://arxiv.org/abs/2412.16339) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- SafeChain: Safety of Language Models with Long Chain-of-Thought Reasoning Capabilities [[Paper]](https://arxiv.org/abs/2502.12025) ![](https://img.shields.io/badge/arXiv-2025.02-red)




### 4.7 Application VII: RAG

- Inference Scaling for Long-Context Retrieval Augmented Generation [[Paper]](https://arxiv.org/abs/2410.04343) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- Plan*RAG: Efficient Test-Time Planning for Retrieval Augmented Generation [[Paper]](https://arxiv.org/abs/2410.20753) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language Models [[Paper]](https://arxiv.org/abs/2411.19443) ![](https://img.shields.io/badge/arXiv-2024.11-red)
- Search-o1: Agentic Search-Enhanced Large Reasoning Models [[Paper]](https://arxiv.org/abs/2501.05366) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- AirRAG: Activating Intrinsic Reasoning for Retrieval Augmented Generation using Tree-based Search [[Paper]](https://arxiv.org/abs/2501.10053) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Chain-of-Retrieval Augmented Generation [[Paper]](https://arxiv.org/abs/2501.14342) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- DeepRAG: Thinking to Retrieval Step by Step for Large Language Models [[Paper]](https://arxiv.org/abs/2502.01142) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning [[Paper]](https://arxiv.org/abs/2503.05592) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning [[Paper]](https://arxiv.org/abs/2503.09516) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning [[Paper]](https://arxiv.org/abs/2503.19470) ![](https://img.shields.io/badge/arXiv-2025.03-red)




### 4.8 Application VIII: Evaluation


- FactScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation [[Paper]](https://aclanthology.org/2023.emnlp-main.741) ![](https://img.shields.io/badge/EMNLP-2023-blue)
- FacTool: Factuality Detection in Generative AI -- A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios [[Paper]](https://arxiv.org/abs/2307.13528) ![](https://img.shields.io/badge/arXiv-2023.07-red)
- Knowledge-Centric Hallucination Detection [[Paper]](https://aclanthology.org/2024.emnlp-main.395/) ![](https://img.shields.io/badge/EMNLP-2024-blue)
- RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation [[Paper]](https://openreview.net/forum?id=J9oefdGUuM) ![](https://img.shields.io/badge/NeurIPS-2024-blue)
- Agent-as-a-Judge: Evaluate Agents with Agents [[Paper]](https://arxiv.org/abs/2410.10934) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge [[Paper]](https://arxiv.org/abs/2501.18099) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Scaling Evaluation-time Compute with Reasoning Models as Process Evaluators [[Paper]](https://arxiv.org/abs/2503.19877) ![](https://img.shields.io/badge/arXiv-2025.03-red)



### 4.9 Special Topic: Representational Complexity of Transformers with CoT 🔥

- The Expressive Power of Transformers with Chain of Thought [[Paper]](https://arxiv.org/abs/2310.07923) ![](https://img.shields.io/badge/arXiv-2023.10-red)
- On the Representational Capacity of Neural Language Models with Chain-of-Thought Reasoning [[Paper]](https://arxiv.org/abs/2406.14197) ![](https://img.shields.io/badge/arXiv-2024.06-red)
- Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought[[Paper]](https://arxiv.org/abs/2501.04682) ![](https://img.shields.io/badge/arXiv-2025.01-red)






## 5. Scaling Reinforcement Learning for Long CoT 🔥


### 5.1 Application: Math & Code


- OpenAI O1 System Card [[Blog]](https://openai.com/index/openai-o1-system-card/) ![](https://img.shields.io/badge/Blog-2024-red)
- QwQ: Reflect Deeply on the Boundaries of the Unknown [[Blog]](https://qwenlm.github.io/blog/qwq-32b-preview/) ![](https://img.shields.io/badge/Blog-2024-red)
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning [[Paper]](https://arxiv.org/abs/2501.12948) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Kimi k1.5: Scaling reinforcement learning with llms [[Paper]](https://arxiv.org/abs/2501.12599) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling [[Paper]](https://arxiv.org/abs/2501.11651) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- There May Not be Aha Moment in R1-Zero-like Training — A Pilot Study [[Blog]](https://oatllm.notion.site/oat-zero) ![](https://img.shields.io/badge/Notion-2025-red)
- 7B Model and 8K Examples: Emerging Reasoning with Reinforcement Learning is Both Effective and Efficient [[Blog]](https://hkust-nlp.notion.site/simplerl-reason) ![](https://img.shields.io/badge/Notion-2025-red)
- Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning [[Paper]](https://arxiv.org/abs/2502.14768) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Demystifying Long Chain-of-Thought Reasoning in LLMs [[Paper]](https://arxiv.org/abs/2502.03373) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution [[Paper]](https://arxiv.org/abs/2502.18449) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- LIMR: Less is More for RL Scaling [[Paper]](https://arxiv.org/abs/2502.11886) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale [[Paper]](https://arxiv.org/abs/2503.14476) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- An Empirical Study on Eliciting and Improving R1-like Reasoning Models [[Paper]](https://arxiv.org/abs/2503.04548) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Open-Reasoner-Zero: An Open Source Approach to Scaling Reinforcement Learning on the Base Model [[Github]](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) ![](https://img.shields.io/badge/GitHub-2025-red)
- DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL [[Blog]](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) ![](https://img.shields.io/badge/Notion-2025-blue)
- Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs [[Paper]](https://arxiv.org/abs/2503.01307) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Understanding R1-Zero-Like Training: A Critical Perspective [[Paper]](https://arxiv.org/abs/2503.20783) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Can Better Cold-Start Strategies Improve RL Training for LLMs? [[Blog]](https://tangible-polo-203.notion.site/Can-Better-Cold-Start-Strategies-Improve-RL-Training-for-LLMs-17aa0742a51680828616c867ed53bc6b) ![](https://img.shields.io/badge/Notion-2025-blue)
- What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret [[Paper]](https://arxiv.org/abs/2503.01491) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't [[Paper]](https://arxiv.org/abs/2503.16219) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning [[Paper]](https://arxiv.org/abs/2504.02546) ![](https://img.shields.io/badge/arXiv-2025.04-red)
- VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks [[Paper]](https://arxiv.org/abs/2504.05118) ![](https://img.shields.io/badge/arXiv-2025.04-red)
- Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining [[Paper]](https://arxiv.org/abs/2504.07912) ![](https://img.shields.io/badge/arXiv-2025.04-red)
- Rethinking Reflection in Pre-Training [[Paper]](https://arxiv.org/abs/2504.04022) ![](https://img.shields.io/badge/arXiv-2025.04-red)




### 5.2 Application: Search

- Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning [[Paper]](https://arxiv.org/abs/2503.09516) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning [[Paper]](https://arxiv.org/abs/2503.05592) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning [[Paper]](https://arxiv.org/abs/2503.09516) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning [[Paper]](https://arxiv.org/abs/2503.19470) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments [[Paper]](https://arxiv.org/abs/2504.03160) ![](https://img.shields.io/badge/arXiv-2025.04-red)

### 5.3 Application III: Multimodal

- QVQ: To See the World with Wisdom [[Blog]](https://qwenlm.github.io/blog/qvq-72b-preview/) ![](https://img.shields.io/badge/Blog-2024-red)
- Llava-onevision: Easy visual task transfer [[Paper]](https://arxiv.org/abs/2408.03326) ![](https://img.shields.io/badge/arXiv-2024.08-red)
- Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution [[Paper]](https://arxiv.org/abs/2409.12191) ![](https://img.shields.io/badge/arXiv-2024.09-red)
- Reducing hallucinations in vision-language models via latent space steering [[Paper]](https://arxiv.org/abs/2410.15778) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- Imagine while Reasoning in Space: Multimodal Visualization-of-Thought [[Paper]](https://arxiv.org/abs/2501.07542) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Kimi k1.5: Scaling reinforcement learning with llms [[Paper]](https://arxiv.org/abs/2501.12599) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Qwen2.5-VL Technical Report [[Paper]](https://arxiv.org/abs/2502.13923) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework [[Github]](https://github.com/hiyouga/EasyR1) ![](https://img.shields.io/badge/GitHub-2025-red)
- LMM-R1 [[Github]](https://github.com/TideDra/lmm-r1) ![](https://img.shields.io/badge/GitHub-2025-red)
- VLM-R1: A stable and generalizable R1-style Large Vision-Language Model [[Github]](https://github.com/om-ai-lab/VLM-R1) ![](https://img.shields.io/badge/GitHub-2025-red)
- R1-V: Reinforcing Super Generalization Ability in Vision-Language Models with Less Than \$3 [[Github]](https://github.com/Deep-Agent/R1-V) ![](https://img.shields.io/badge/GitHub-2025-red)
- open-r1-multimodal: A fork to add multimodal model training to open-r1 [[Github]](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) ![](https://img.shields.io/badge/GitHub-2025-red)
- R1-Multimodal-Journey: A journey to real multimodal R1 [[Github]](https://github.com/FanqingM/R1-Multimodal-Journey) ![](https://img.shields.io/badge/GitHub-2025-red)
- Open-R1-Video [[Github]](https://github.com/Wang-Xiaodong1899/Open-R1-Video) ![](https://img.shields.io/badge/GitHub-2025-red)
- Video-R1: Reinforcing Video Reasoning in MLLMs [[Paper]](https://arxiv.org/abs/2503.21776) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model [[Paper]](https://arxiv.org/abs/2503.05132) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- MM-Eureka: Exploring the Frontiers of Multimodal Reasoning with Rule-based Reinforcement Learning [[Paper]](https://arxiv.org/abs/2503.07365) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement [[Paper]](https://arxiv.org/abs/2503.06520) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Rethinking RL Scaling for Vision Language Models: A Transparent, From-Scratch Framework and Comprehensive Evaluation Scheme [[Paper]](https://arxiv.org/abs/2504.02587) ![](https://img.shields.io/badge/arXiv-2025.04-red)
- Improved Visual-Spatial Reasoning via R1-Zero-Like Training [[Paper]](https://arxiv.org/abs/2504.00883) ![](https://img.shields.io/badge/arXiv-2025.04-red)
- Kimi-VL [[Blog]](https://github.com/MoonshotAI/Kimi-VL) ![](https://img.shields.io/badge/Blog-2025-red)
- Introducing OpenAI o3 and o4-mini [[Blog]](https://openai.com/index/introducing-o3-and-o4-mini/) ![](https://img.shields.io/badge/Blog-2025-red)




### 5.4 Papers Sorted by RL Components

#### 5.4.1 Training Algorithm
- Proximal Policy Optimization Algorithms [[Paper]](https://arxiv.org/abs/1707.06347) ![](https://img.shields.io/badge/arXiv-2017.07-red)
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models [[Paper]](https://arxiv.org/abs/2402.03300) ![](https://img.shields.io/badge/arXiv-2024.02-red)
- REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models [[Paper]](https://arxiv.org/abs/2501.03262) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning [[Paper]](https://arxiv.org/abs/2502.06781) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale [[Paper]](https://arxiv.org/abs/2503.14476) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Understanding R1-Zero-Like Training: A Critical Perspective [[Paper]](https://arxiv.org/abs/2503.20783) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret [[Paper]](https://arxiv.org/abs/2503.01491) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning [[Paper]](https://arxiv.org/abs/2504.02546) ![](https://img.shields.io/badge/arXiv-2025.04-red)
- VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks [[Paper]](https://arxiv.org/abs/2504.05118) ![](https://img.shields.io/badge/arXiv-2025.04-red)


#### 5.4.2 Reward Model


- Self-Explore: Enhancing Mathematical Reasoning in Language Models with Fine-grained Rewards [[Paper]](https://arxiv.org/abs/2404.10346) ![](https://img.shields.io/badge/arXiv-2024.04-red)
- Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms [[Paper]](https://arxiv.org/abs/2406.02900) ![](https://img.shields.io/badge/arXiv-2024.06-red)
- On Designing Effective RL Reward at Training Time for LLM Reasoning [[Paper]](https://arxiv.org/abs/2410.15115) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment [[Paper]](https://arxiv.org/abs/2410.01679) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- Does RLHF Scale? Exploring the Impacts From Data, Model, and Method [[Paper]](https://arxiv.org/abs/2412.06000) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- Process Reinforcement through Implicit Rewards [[Paper]](https://arxiv.org/abs/2502.01456) ![](https://img.shields.io/badge/arXiv-2025.02-red)

#### 5.4.3 Base Model

- Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs [[Paper]](https://arxiv.org/abs/2503.01307) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Understanding R1-Zero-Like Training: A Critical Perspective [[Paper]](https://arxiv.org/abs/2503.20783) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining [[Paper]](https://arxiv.org/abs/2504.07912) ![](https://img.shields.io/badge/arXiv-2025.04-red)
- Rethinking Reflection in Pre-Training [[Paper]](https://arxiv.org/abs/2504.04022) ![](https://img.shields.io/badge/arXiv-2025.04-red)



#### 5.4.4 Training Data
- Kimi k1.5: Scaling reinforcement learning with llms [[Paper]](https://arxiv.org/abs/2501.12599) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- LIMR: Less is More for RL Scaling [[Paper]](https://arxiv.org/abs/2502.11886) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale [[Paper]](https://arxiv.org/abs/2503.14476) ![](https://img.shields.io/badge/arXiv-2025.03-red)

#### 5.4.5 Multi-stage Training
- Kimi k1.5: Scaling reinforcement learning with llms [[Paper]](https://arxiv.org/abs/2501.12599) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Demystifying Long Chain-of-Thought Reasoning in LLMs [[Paper]](https://arxiv.org/abs/2502.03373) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning [[Paper]](https://arxiv.org/abs/2502.14768) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL [[Blog]](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) ![](https://img.shields.io/badge/Notion-2025-blue)
- Can Better Cold-Start Strategies Improve RL Training for LLMs? [[Blog]](https://tangible-polo-203.notion.site/Can-Better-Cold-Start-Strategies-Improve-RL-Training-for-LLMs-17aa0742a51680828616c867ed53bc6b) ![](https://img.shields.io/badge/Notion-2025-blue)


#### 5.4.6 Evaluation

- Position: Benchmarking is Limited in Reinforcement Learning Research [[Paper]](https://arxiv.org/abs/2406.16241) ![](https://img.shields.io/badge/ICLR-2024-blue)
- A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility [[Paper]](https://arxiv.org/abs/2504.07086) ![](https://img.shields.io/badge/arXiv-2025.04-red)




#### 5.4.7 Analysis

- Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data [[Paper]](https://arxiv.org/abs/2404.14367) ![](https://img.shields.io/badge/arXiv-2024.04-red)
- RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold [[Paper]](https://arxiv.org/abs/2406.14532) ![](https://img.shields.io/badge/arXiv-2024.06-red)



### 5.5 Infra



[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) ![GitHub stars](https://img.shields.io/github/stars/OpenRLHF/OpenRLHF)

[Verl](https://github.com/volcengine/verl) ![GitHub stars](https://img.shields.io/github/stars/volcengine/verl)

[NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/NeMo-Aligner)

[Deepspeed-chat](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-chat) ![GitHub stars](https://img.shields.io/github/stars/deepspeedai/DeepSpeed)


## 6. Supervised Learning for Long CoT 🔥

### 6.1 Long CoT Resource


| **Work** | **Application** | **Type** | **Source** | **Quantity** | **Modality** | **Link** |
|----------|-----------------|----------|------------|--------------|--------------|----------|
| O1 Journey–Part 1 | Math | Synthesize | GPT-4o | 0.3K | Text | [GitHub](https://github.com/GAIR-NLP/O1-Journey) [HuggingFace](https://huggingface.co/datasets/GAIR/o1-journey) |
| Marco-o1 | Reasoning | Synthesize | Qwen2-7B-Instruct | 10K | Text | [GitHub](https://github.com/AIDC-AI/Marco-o1) |
| STILL-2 | Math, Code, Science, Puzzle | Distillation | DeepSeek-R1-Lite-Preview, QwQ-32B-preview | 5K | Text | [GitHub](https://github.com/RUCAIBox/Slow_Thinking_with_LLMs) [HuggingFace](https://huggingface.co/datasets/RUC-AIBOX/long_form_thought_data_5k) |
| RedStar-math | Math | Distillation | QwQ-32B-preview | 4K | Text | [HuggingFace](https://huggingface.co/datasets/RedStar-Reasoning/math_dataset) |
| RedStar-code | Code | Distillation | QwQ-32B-preview | 16K | Text | [HuggingFace](https://huggingface.co/datasets/RedStar-Reasoning/code_dataset) |
| RedStar-multimodal | Math | Distillation | QwQ-32B-preview | 12K | Vision + Text | [HuggingFace](https://huggingface.co/datasets/RedStar-Reasoning/multimodal_dataset) |
| S1K | Math, Science, Code | Distillation | Gemini Flash Thinking | 1K | Text | [GitHub](https://github.com/simplescaling/s1) [HuggingFace](https://huggingface.co/datasets/simplescaling/s1K) |
| S1K-1.1 | Math, Science, Code | Distillation | DeepSeek R1 | 1K | Text | [GitHub](https://github.com/simplescaling/s1) [HuggingFace](https://huggingface.co/datasets/simplescaling/s1K-1.1) |
| LIMO | Math | Distillation | DeepSeek R1, DeepSeekR1-Distill-Qwen-32B | 0.8K | Text | [GitHub](https://github.com/GAIR-NLP/LIMO) [HuggingFace](https://huggingface.co/datasets/GAIR/LIMO) |
| OpenThoughts-114k | Math, Code, Science, Puzzle | Distillation | DeepSeek R1 | 114K | Text | [GitHub](https://github.com/open-thoughts/open-thoughts) [HuggingFace](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |
| OpenR1-Math-220k | Math | Distillation | DeepSeek R1 | 220K | Text | [GitHub](https://github.com/huggingface/open-r1) [HuggingFace](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) |
| OpenThoughts2-1M | Math, Code, Science, Puzzle | Distillation | DeepSeek R1 | 1M | Text | [GitHub](https://github.com/open-thoughts/open-thoughts) [HuggingFace](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M) |
| CodeForces-CoTs | Code | Distillation | DeepSeek R1 | 47K | Text | [GitHub](https://github.com/huggingface/open-r1) [HuggingFace](https://huggingface.co/datasets/open-r1/codeforces-cots) |
| Sky-T1-17k | Math, Code, Science, Puzzle | Distillation | QwQ-32B-Preview | 17K | Text | [GitHub](https://github.com/NovaSky-AI/SkyThought) [HuggingFace](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k) |
| S²R | Math | Synthesize | Qwen2.5-Math-7B | 3K | Text | [GitHub](https://github.com/NineAbyss/S2R) [HuggingFace](https://huggingface.co/datasets/S2R-data/S2R-dataset) |
| R1-Onevision | Science, Math, General | Distillation | DeepSeek R1 | 155K | Vision + Text | [GitHub](https://github.com/Fancy-MLLM/R1-Onevision) [HuggingFace](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision) |
| OpenO1-SFT | Math, Code | Synthesize | - | 77K | Text | [GitHub](https://github.com/Open-Source-O1/Open-O1) [HuggingFace](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT) |
| Medical-o1 | Medical | Distillation | Deepseek R1 | 25K | Text | [GitHub](https://github.com/FreedomIntelligence/HuatuoGPT-o1) [HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) |
| O1 Journey–Part 3 | Medical | Distillation | o1-preview | 0.5K | Text | [GitHub](https://github.com/SPIRAL-MED/Ophiuchus) [HuggingFace](https://huggingface.co/datasets/SPIRAL-MED/o1-journey-Ophiuchus) |
| SCP-116K | Math, Science | Distillation | Deepseek R1 | 116K | Text | [GitHub](https://github.com/AQA6666/SCP-116K-open) [HuggingFace](https://huggingface.co/datasets/EricLu/SCP-116K) |
| open-r1-multimodal | Math | Distillation | GPT-4o | 8K | Vision + Text | [GitHub](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) [HuggingFace](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) |
| Vision-R1-cold | Science, Math, General | Distillation | Deepseek R1 | 200K | Vision + Text | [GitHub](https://github.com/Osilly/Vision-R1) [HuggingFace](https://huggingface.co/datasets/Osilly/Vision-R1-cold) |
| MMMU-Reasoning-Distill-Validation | Science, Math, General | Distillation | Deepseek R1 | 0.8K | Vision + Text | [ModelScope](https://www.modelscope.cn/datasets/modelscope/MMMU-Reasoning-Distill-Validation) |
| Clevr-CoGenT | Vision Counting | Distillation | Deepseek R1 | 37.8K | Vision + Text | [GitHub](https://github.com/Deep-Agent/R1-V) [HuggingFace](https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_TrainA_R1) |
| VL-Thinking | Science, Math, General | Distillation | Deepseek R1 | 158K | Vision + Text | [GitHub](https://github.com/UCSC-VLAA/VL-Thinking) [HuggingFace](https://huggingface.co/datasets/UCSC-VLAA/VL-Thinking) |
| Video-R1 | Video | Distillation | Qwen2.5-VL-72B | 158K | Vision + Text | [GitHub](https://github.com/tulerfeng/Video-R1) [HuggingFace](https://huggingface.co/datasets/Video-R1/Video-R1-data) |
| Embodied-Reasoner | Embodied AI | Synthesize | GPT-4o | 9K | Vision + Text | [GitHub](https://github.com/zwq2018/embodied_reasoner) [HuggingFace](https://huggingface.co/datasets/zwq2018/embodied_reasoner) |
| OpenCodeReasoning | Code | Distillation | DeepSeek R1 | 736K | Text | [HuggingFace](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) |
| SafeChain | Safety | Distillation | Deepseek R1 | 40K | Text | [GitHub](https://github.com/uw-nsl/safechain) [HuggingFace](https://huggingface.co/datasets/UWNSL/SafeChain) |
| KodCode | Code | Distillation | DeepSeek R1 | 2.8K | Text | [GitHub](https://github.com/KodCode-AI/kodcode) [HuggingFace](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1) |




### 6.2 Analysis

- Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping [[Paper]](https://arxiv.org/abs/2402.14083) ![](https://img.shields.io/badge/arXiv-2024.02-red)
- Stream of Search (SoS): Learning to Search in Language [[Paper]](https://arxiv.org/abs/2404.03683) ![](https://img.shields.io/badge/arXiv-2024.04-red)
- Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems [[Paper]](https://arxiv.org/abs/2408.16293) ![](https://img.shields.io/badge/arXiv-2024.08-red)
- Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems [[Paper]](https://arxiv.org/abs/2412.09413) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- s1: Simple test-time scaling [[Paper]](https://arxiv.org/abs/2501.19393) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- RedStar: Does Scaling Long-CoT Data Unlock Better Slow-Reasoning Systems? [[Paper]](https://arxiv.org/abs/2501.11284) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training [[Paper]](https://arxiv.org/abs/2501.17161) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- LIMO: Less is More for Reasoning [[Paper]](https://arxiv.org/abs/2502.03387) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Small Models Struggle to Learn from Strong Reasoners [[Paper]](https://arxiv.org/abs/2502.12143) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters! [[Paper]](https://arxiv.org/abs/2502.07374) ![](https://img.shields.io/badge/arXiv-2025.02-red)


## 7. Self-improvement with Test-Time Scaling

### 7.1 Parallel Sampling


- STaR: Bootstrapping Reasoning With Reasoning [[Paper]](http://papers.nips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2022-blue)
- RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment [[Paper]](https://arxiv.org/abs/2304.06767) ![](https://img.shields.io/badge/arXiv-2023.04-red)
- Language Models Can Teach Themselves to Program Better [[Paper]](https://openreview.net/pdf?id=SaRj2ka1XZ3) ![](https://img.shields.io/badge/ICLR-2023-blue)
- Scaling relationship on learning mathematical reasoning with large language models [[Paper]](https://arxiv.org/abs/2308.01825) ![](https://img.shields.io/badge/arXiv-2023.08-red)
- Reinforced self-training (rest) for language modeling [[Paper]](https://arxiv.org/abs/2308.08998) ![](https://img.shields.io/badge/arXiv-2023.08-red)
- Large Language Models Can Self-Improve [[Paper]](https://aclanthology.org/2023.emnlp-main.67) ![](https://img.shields.io/badge/EMNLP-2023-blue)
- Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models [[Paper]](https://arxiv.org/abs/2312.06585) ![](https://img.shields.io/badge/arXiv-2023.12-red)
- Self-Rewarding Language Models [[Paper]](https://arxiv.org/abs/2401.10020) ![](https://img.shields.io/badge/arXiv-2024.01-red)
- V-STaR: Training Verifiers for Self-Taught Reasoners [[Paper]](https://arxiv.org/abs/2402.06457) ![](https://img.shields.io/badge/arXiv-2024.02-red)
- Iterative Reasoning Preference Optimization [[Paper]](https://arxiv.org/abs/2404.19733) ![](https://img.shields.io/badge/arXiv-2024.04-red)
- Progress or Regress? Self-Improvement Reversal in Post-training [[Paper]](https://arxiv.org/abs/2407.05013) ![](https://img.shields.io/badge/arXiv-2024.07-red)
- Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement [[Paper]](https://arxiv.org/abs/2409.12122) ![](https://img.shields.io/badge/arXiv-2024.09-red)
- Process-based Self-Rewarding Language Models [[Paper]](https://arxiv.org/abs/2503.03746) ![](https://img.shields.io/badge/arXiv-2025.03-red)
### 7.2 Tree Search

- Alphazero-like Tree-Search Can Guide Large Language Model Decoding and Training [[Paper]](https://arxiv.org/abs/2309.17179) ![](https://img.shields.io/badge/arXiv-2023.09-red)  
- AlphaMath Almost Zero: Process Supervision without Process [[Paper]](https://arxiv.org/abs/2405.03553) ![](https://img.shields.io/badge/arXiv-2024.05-red)  
- Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning [[Paper]](https://arxiv.org/abs/2405.00451) ![](https://img.shields.io/badge/arXiv-2024.05-red)  
- ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search [[Paper]](https://arxiv.org/abs/2406.03816) ![](https://img.shields.io/badge/arXiv-2024.06-red)  
- Step-Level Value Preference Optimization for Mathematical Reasoning [[Paper]](https://arxiv.org/abs/2406.10858) ![](https://img.shields.io/badge/arXiv-2024.06-red)  
- Chain of Preference Optimization: Improving Chain-of-Thought Reasoning in LLMs [[Paper]](https://arxiv.org/abs/2406.09136) ![](https://img.shields.io/badge/arXiv-2024.06-red)  
- rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking [[Paper]](https://arxiv.org/abs/2501.04519) ![](https://img.shields.io/badge/arXiv-2025.01-red)  





### 7.3 Multi-turn Correction
- Multi-Turn Code Generation Through Single-Step Rewards [[Paper]](https://arxiv.org/abs/2502.20380) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- S$^2$R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning [[Paper]](https://arxiv.org/abs/2502.12853) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Self-rewarding correction for mathematical reasoning [[Paper]](https://arxiv.org/abs/2502.19613) ![](https://img.shields.io/badge/arXiv-2025.02-red)

### 7.4 Long CoT
- Thinking Longer, Not Larger: Enhancing Software Engineering Agents via Scaling Test-Time Compute [[Paper]](https://arxiv.org/abs/2503.23803) ![](https://img.shields.io/badge/arXiv-2025.03-red)


## 8. Ensemble of Test-Time Scaling Method


- Accessing GPT-4 Level Mathematical Olympiad Solutions via Monte Carlo Tree Self-Refine with LLaMa-3 8B [[Paper]](https://arxiv.org/abs/2406.07394) ![](https://img.shields.io/badge/arXiv-2024.06-red)
- Scaling test-time compute with open models [[Blog]](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) ![](https://img.shields.io/badge/HuggingFace-2024-blue)
- Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters [[Paper]](https://arxiv.org/abs/2408.03314) ![](https://img.shields.io/badge/arXiv-2024.08-red)
- RethinkMCTS: Refining Erroneous Thoughts in Monte Carlo Tree Search for Code Generation [[Paper]](https://arxiv.org/abs/2409.09584) ![](https://img.shields.io/badge/arXiv-2024.09-red)
- TreeBoN: Enhancing Inference-Time Alignment with Speculative Tree-Search and Best-of-N Sampling [[Paper]](https://arxiv.org/abs/2410.16033) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- LLaMA-Berry: Pairwise Optimization for O1-like Olympiad-Level Mathematical Reasoning [[Paper]](https://arxiv.org/abs/2410.02884) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- MC-NEST -- Enhancing Mathematical Reasoning in Large Language Models with a Monte Carlo Nash Equilibrium Self-Refine Tree [[Paper]](https://arxiv.org/abs/2411.15645) ![](https://img.shields.io/badge/arXiv-2024.11-red)
- SPaR: Self-Play with Tree-Search Refinement to Improve Instruction-Following in Large Language Models [[Paper]](https://arxiv.org/abs/2412.11605) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning [[Paper]](https://arxiv.org/abs/2412.09078) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- RealCritic: Towards Effectiveness-Driven Evaluation of Language Model Critiques [[Paper]](https://arxiv.org/abs/2501.14492) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling [[Paper]](https://arxiv.org/abs/2502.06703) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Scaling Test-Time Compute Without Verification or RL is Suboptimal [[Paper]](https://arxiv.org/abs/2502.12118) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- S*: Test Time Scaling for Code Generation [[Paper]](https://arxiv.org/abs/2502.14382) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Thinking Longer, Not Larger: Enhancing Software Engineering Agents via Scaling Test-Time Compute [[Paper]](https://arxiv.org/abs/2503.23803) ![](https://img.shields.io/badge/arXiv-2025.03-red)

## 9. Inference Time Scaling Laws


- The Impact of Reasoning Step Length on Large Language Models [[Paper]](https://arxiv.org/abs/2401.04925) ![](https://img.shields.io/badge/arXiv-2024.01-red)
- More Agents Is All You Need [[Paper]](https://arxiv.org/abs/2402.05120) ![](https://img.shields.io/badge/arXiv-2024.02-red)
- Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems [[Paper]](https://arxiv.org/abs/2403.02419) ![](https://img.shields.io/badge/arXiv-2024.03-red)
- Large Language Monkeys: Scaling Inference Compute with Repeated Sampling [[Paper]](https://arxiv.org/abs/2407.21787) ![](https://img.shields.io/badge/arXiv-2024.07-red)
- Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models [[Paper]](https://arxiv.org/abs/2408.00724) ![](https://img.shields.io/badge/arXiv-2024.08-red)
- Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling [[Paper]](https://arxiv.org/abs/2408.16737) ![](https://img.shields.io/badge/arXiv-2024.08-red)
- Inference Scaling fLaws: The Limits of LLM Resampling with Imperfect Verifiers [[Paper]](https://arxiv.org/abs/2411.17501) ![](https://img.shields.io/badge/arXiv-2024.11-red)
- s1: Simple test-time scaling [[Paper]](https://arxiv.org/abs/2501.19393) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling [[Paper]](https://arxiv.org/abs/2501.11651) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities? [[Paper]](https://arxiv.org/abs/2502.12215) ![](https://img.shields.io/badge/arXiv-2025.02-red)


## 10. Improving Scaling Efficiency

### 10.1 Parallel Sampling


- Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs [[Paper]](https://aclanthology.org/2023.emnlp-main.761) ![](https://img.shields.io/badge/EMNLP-2023-blue)
- Universal Self-Consistency for Large Language Model Generation [[Paper]](https://arxiv.org/abs/2311.17311) ![](https://img.shields.io/badge/arXiv-2023.11-red)
- Escape Sky-High Cost: Early-Stopping Self-Consistency for Multi-Step Reasoning [[Paper]](https://arxiv.org/abs/2401.10480) ![](https://img.shields.io/badge/arXiv-2024.01-red)
- Easy-to-Hard Generalization: Scalable Alignment Beyond Human Supervision [[Paper]](https://arxiv.org/abs/2403.09472) ![](https://img.shields.io/badge/arXiv-2024.03-red)
- Dynamic Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling [[Paper]](https://arxiv.org/abs/2408.17017) ![](https://img.shields.io/badge/arXiv-2024.08-red)
- Make Every Penny Count: Difficulty-Adaptive Self-Consistency for Cost-Efficient Reasoning [[Paper]](https://arxiv.org/abs/2408.13457) ![](https://img.shields.io/badge/arXiv-2024.08-red)
- Fast Best-of-N Decoding via Speculative Rejection [[Paper]](https://arxiv.org/abs/2410.20290) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models [[Paper]](https://arxiv.org/abs/2412.15287) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers [[Paper]](https://arxiv.org/abs/2502.20379) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Scalable Best-of-N Selection for Large Language Models via Self-Certainty [[Paper]](https://arxiv.org/abs/2502.18581) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding [[Paper]](https://arxiv.org/abs/2503.01422) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning [[Paper]](https://arxiv.org/abs/2504.01005) ![](https://img.shields.io/badge/arXiv-2025.04-red)


### 10.2 Tree Search

- LiteSearch: Efficacious Tree Search for LLM [[Paper]](https://arxiv.org/abs/2407.00320) ![](https://img.shields.io/badge/arXiv-2024.07-red)
- ETS: Efficient Tree Search for Inference-Time Scaling [[Paper]](https://arxiv.org/abs/2502.13575) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Dynamic Parallel Tree Search for Efficient LLM Reasoning [[Paper]](https://arxiv.org/abs/2502.16235) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Don't Get Lost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls [[Paper]](https://arxiv.org/abs/2502.11183) ![](https://img.shields.io/badge/arXiv-2025.02-red)


### 10.3 Multi-turn Correction

- Can Large Language Models Be an Alternative to Human Evaluations? [[Paper]](https://aclanthology.org/2023.acl-long.870) ![](https://img.shields.io/badge/ACL-2023-blue)
- G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment [[Paper]](https://aclanthology.org/2023.emnlp-main.153) ![](https://img.shields.io/badge/EMNLP-2023-blue)
- A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-Confidence Generation [[Paper]](https://arxiv.org/abs/2307.03987) ![](https://img.shields.io/badge/arXiv-2023.07-red)
- Confidence Matters: Revisiting Intrinsic Self-Correction Capabilities of Large Language Models [[Paper]](https://arxiv.org/abs/2402.12563) ![](https://img.shields.io/badge/arXiv-2024.02-red)
- Branch-Solve-Merge Improves Large Language Model Evaluation and Generation [[Paper]](https://aclanthology.org/2024.naacl-long.462) ![](https://img.shields.io/badge/NAACL-2024-blue)
- Recursive Introspection: Teaching Language Model Agents How to Self-Improve [[Paper]](https://arxiv.org/abs/2407.18219) ![](https://img.shields.io/badge/arXiv-2024.07-red)
- Training Language Models to Self-Correct via Reinforcement Learning [[Paper]](https://arxiv.org/abs/2409.12917) ![](https://img.shields.io/badge/arXiv-2024.09-red)
- Enhancing LLM Reasoning via Critique Models with Test-Time and Training-Time Supervision [[Paper]](https://arxiv.org/abs/2411.16579) ![](https://img.shields.io/badge/arXiv-2024.11-red)


### 10.4 Long CoT


- Implicit Chain of Thought Reasoning via Knowledge Distillation [[Paper]](https://arxiv.org/abs/2311.01460) ![](https://img.shields.io/badge/arXiv-2023.11-red)
- Anchor-based Large Language Models [[Paper]](https://arxiv.org/abs/2402.07616) ![](https://img.shields.io/badge/arXiv-2024.02-red)
- From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step [[Paper]](https://arxiv.org/abs/2405.14838) ![](https://img.shields.io/badge/arXiv-2024.05-red)
- Break the Chain: Large Language Models Can be Shortcut Reasoners [[Paper]](https://arxiv.org/abs/2406.06580) ![](https://img.shields.io/badge/arXiv-2024.06-red)
- Distilling System 2 into System 1 [[Paper]](https://arxiv.org/abs/2407.06023) ![](https://img.shields.io/badge/arXiv-2024.07-red)
- System-1.x: Learning to Balance Fast and Slow Planning with Language Models [[Paper]](https://arxiv.org/abs/2407.14414) ![](https://img.shields.io/badge/arXiv-2024.07-red)
- Concise Thoughts: Impact of Output Length on LLM Reasoning and Cost [[Paper]](https://arxiv.org/abs/2407.19825) ![](https://img.shields.io/badge/arXiv-2024.07-red)
- To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning [[Paper]](https://arxiv.org/abs/2409.12183) ![](https://img.shields.io/badge/arXiv-2024.09-red)
- Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces [[Paper]](https://arxiv.org/abs/2410.09918) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- Markov Chain of Thought for Efficient Mathematical Reasoning [[Paper]](https://arxiv.org/abs/2410.17635) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- Can Language Models Learn to Skip Steps? [[Paper]](https://arxiv.org/abs/2411.01855) ![](https://img.shields.io/badge/arXiv-2024.11-red)
- Training Large Language Models to Reason in a Continuous Latent Space [[Paper]](https://arxiv.org/abs/2412.06769) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- Compressed Chain of Thought: Efficient Reasoning Through Dense Representations [[Paper]](https://arxiv.org/abs/2412.13171) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- C3oT: Generating Shorter Chain-of-Thought without Compromising Effectiveness [[Paper]](https://arxiv.org/abs/2412.11664) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- Token-Budget-Aware LLM Reasoning [[Paper]](https://arxiv.org/abs/2412.18547) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- Efficiently Serving LLM Reasoning Programs with Certaindex [[Paper]](https://arxiv.org/abs/2412.20993) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs [[Paper]](https://arxiv.org/abs/2412.21187) ![](https://img.shields.io/badge/arXiv-2024.12-red)
- Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models [[Paper]](https://openreview.net/forum?id=DWkJCSxKU5) ![](https://img.shields.io/badge/TMLR-2024-blue)
- O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning [[Paper]](https://arxiv.org/abs/2501.12570) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Think Smarter not Harder: Adaptive Reasoning with Inference Aware Optimization [[Paper]](https://arxiv.org/abs/2501.17974) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs [[Paper]](https://arxiv.org/abs/2501.18585) ![](https://img.shields.io/badge/arXiv-2025.01-red)
- Training Language Models to Reason Efficiently [[Paper]](https://arxiv.org/abs/2502.04463) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach [[Paper]](https://arxiv.org/abs/2502.05171) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- CoT-Valve: Length-Compressible Chain-of-Thought Tuning [[Paper]](https://arxiv.org/abs/2502.09601) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- TokenSkip: Controllable Chain-of-Thought Compression in LLMs [[Paper]](https://arxiv.org/abs/2502.12067) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs [[Paper]](https://arxiv.org/abs/2502.12134) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models [[Paper]](https://arxiv.org/abs/2502.13260) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- The Relationship Between Reasoning and Performance in Large Language Models -- o3 (mini) Thinks Harder, Not Longer [[Paper]](https://arxiv.org/abs/2502.15631) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- LightThinker: Thinking Step-by-Step Compression [[Paper]](https://arxiv.org/abs/2502.15589) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Chain of Draft: Thinking Faster by Writing Less [[Paper]](https://arxiv.org/abs/2502.18600) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning [[Paper]](https://arxiv.org/abs/2502.18080) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Self-Training Elicits Concise Reasoning in Large Language Models [[Paper]](https://arxiv.org/abs/2502.20122) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation [[Paper]](https://arxiv.org/abs/2502.21074) ![](https://img.shields.io/badge/arXiv-2025.02-red)
- Efficient Test-Time Scaling via Self-Calibration [[Paper]](https://arxiv.org/abs/2503.00031) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach [[Paper]](https://arxiv.org/abs/2503.01141) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- DAST: Difficulty-Adaptive Slow-Thinking for Large Reasoning Models [[Paper]](https://arxiv.org/abs/2503.04472) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning [[Paper]](https://arxiv.org/abs/2503.04697) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching [[Paper]](https://arxiv.org/abs/2503.05179) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models [[Paper]](https://arxiv.org/abs/2503.06692) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning [[Paper]](https://arxiv.org/abs/2503.07572) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging [[Paper]](https://arxiv.org/abs/2503.20641) ![](https://img.shields.io/badge/arXiv-2025.03-red)
- Think Less, Achieve More: Cut Reasoning Costs by 50% Without Sacrificing Accuracy [[Blog]](https://novasky-ai.github.io/posts/reduce-overthinking) ![](https://img.shields.io/badge/Blog-2025-blue)



## 11.  Latent Thoughts



- Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking [[Paper]](https://arxiv.org/abs/2403.09629) ![](https://img.shields.io/badge/arXiv-2024.03-red)
- Lean-STaR: Learning to Interleave Thinking and Proving [[Paper]](https://arxiv.org/abs/2407.10040) ![](https://img.shields.io/badge/arXiv-2024.07-red)
- RATIONALYST: Pre-training Process-Supervision for Improving Reasoning [[Paper]](https://arxiv.org/abs/2410.01044) ![](https://img.shields.io/badge/arXiv-2024.10-red)
- Reasoning to Learn from Latent Thoughts [[Paper]](https://arxiv.org/abs/2503.18866) ![](https://img.shields.io/badge/arXiv-2025.03-red)


