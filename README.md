# *Writing Like the Best*: Exemplar-Based Expository Text Generation

---

This repository contains source code and data for our ACL 2025 Paper [*Writing Like the Best*: Exemplar-Based Expository Text Generation](https://arxiv.org/abs/2505.18859).

## How to Run

### Step 1: Fill in your secret keys in "secret_keys.sh".

### Step 2: Run main experiments

You may change the *Wikipedia* to *RoleEE* or *USNews* for corresponding datasets, or change *GPT4* to *LLaMA3* for different backbone LLM, and change *false* to *true* for debug mode. You may change the corresponding config files for detailed settings, which is under the "configs" folder.

> bash experiments.sh Wikipedia GPT4 false

### Step 3: Run Evaluations

You may change *baseline* to *ablation* for different evaluation modes.

> bash evaluations.sh

## Data

## Reference

If you use our work, please consider citing our work:

```
@inproceedings{liu-chang-2025-writing,
    title = "Writing Like the Best: Exemplar-Based Expository Text Generation",
    author = "Liu, Yuxiang and Chang, Kevin Chen-Chuan",
    editor = "Che, Wanxiang and Nabende, Joyce and Shutova, Ekaterina and Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1250/",
    doi = "10.18653/v1/2025.acl-long.1250",
    pages = "25739--25764",
    ISBN = "979-8-89176-251-0",
}
```