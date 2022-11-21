# A Combination of Visual-Semantic Reasoning and Text Entailment-based Boosting Algorithm for Cheapfake Detection

## Requirements based on each step
- Extract image features [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention). Recommend using [docker](https://hub.docker.com/r/airsplay/bottom-up-attention)
- Train image-caption matching model [VSRN](https://github.com/KunpengLi1994/VSRN)
_Note: We recommend using Docker with GPU for easy installation._

### Extract features:
To extract features of image, use bottom-up-attention. Running with following requirement: bottom-up-attention/train.py
Can download via [link](https://drive.google.com/file/d/1Jc8qs5zyXvsthVwRtKzG6ua-8Yacf-wb/view?usp=sharing).

### Train image-caption matching model:
Running with following requirement: VSRN/train.py

### Evaluate on task 1 and task 2 of challenge:
Running with following requirement: ACMMM/acmmm_evaluate_task_1.py & ACMMM/acmmm_evaluate_task_2.py
