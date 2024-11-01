# LLM Fact Memorization Study

## Introduction
This project investigates how Large Language Models (LLMs) learn and memorize new facts, drawing parallels with human learning patterns. While humans typically need to encounter new information around 7 times to achieve reliable memorization, we explore whether similar patterns exist in LLMs.

Our research focuses on:
- Comparing passive learning vs. active exam-based learning approaches
- Analyzing the impact of reinforcement learning on fact memorization
- Investigating factors affecting memorization difficulty:
  - Prior knowledge and exposure
  - Presence of contradictory information in the model's knowledge base
  - Different training techniques (SFT, DPO)

## Repository Structure

```
📦 llm-fact-memorization
├── 📂 algorithms
│   ├── 📜 dpo.py          # Direct Preference Optimization implementation
│   └── 📜 sft.py          # Supervised Fine-Tuning implementation
├── 📂 data
│   ├── 📂 name_number_query    # Dataset for name-number association tasks
│   └── 📂 news_articles        # News article Q&A datasets
├── 📂 evaluation_tools
│   └── 📜 answer_evaluator.py  # Evaluation metrics and tools
├── 📂 experiments
│   ├── 📜 name_number_query_sft.py
│   ├── 📜 name_number_query_dpo.py
│   ├── 📜 news_article_sft.py
│   └── 📜 news_article_dpo.py
├── 📜 config.py               # Configuration classes
├── 📜 requirements.txt        # Package dependencies
└── 📜 README.md
```

## Experiments

### 1. Name-Number Query Task
Tests LLM's ability to memorize associations between names and 10-digit numbers.


#### Run SFT Training
```bash
python experiments/name_number_query_sft.py
```

#### Run DPO Training (after SFT)
```bash
python experiments/name_number_query_dpo.py
```

### 2. News Article Q&A
Evaluates model's ability to learn and recall facts from news articles through Q&A.

#### Run SFT Training
```bash
python experiments/news_article_sft.py
```

#### Run DPO Training (after SFT)
```bash
python experiments/news_article_dpo.py
```

Each experiment includes:
- Supervised Fine-Tuning (SFT) for initial learning
- Direct Preference Optimization (DPO) for refinement
- Comprehensive evaluation metrics
- Detailed logging of training progress

## Requirements

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Required API keys:
- Create `apikeys.json` in the root directory:
```json
{
    "openai_api_key": "",
    "news_api_key": "",
    "hf_api_key": ""
}
```

4. Hardware Requirements:
- GPU with at least 16GB VRAM
- 32GB+ RAM recommended
- 100GB+ storage for model weights and datasets

## License

MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.