# When are Deep Networks really better than Decision Forests at small sample sizes, and how?

[![arXiv](https://img.shields.io/badge/arXiv-2108.13637-red.svg?style=flat)](https://arxiv.org/abs/2108.13637)
[![CircleCI](https://circleci.com/gh/neurodata/df-dn-paper/tree/main.svg?style=shield)](https://circleci.com/gh/neurodata/df-dn-paper/tree/main)
[![Netlify](https://img.shields.io/netlify/e77b134b-1e9b-4ae9-b378-822615333dbd)](https://app.netlify.com/sites/dfdn/deploys)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/licenses/MIT)

**This is preliminary work. More details will be available.**

- **Documentation:** https://dfdn.neurodata.io/
- **Abstract:** https://dfdn.neurodata.io/#abstract
- **Replication Guide:** https://dfdn.neurodata.io/#replicate
- **Benchmark Figures:** https://dfdn.neurodata.io/#benchmarks


## Updates

**DF/DN:** This project aims to compare the performance differences between **D**ecision **F**orests & **D**eep **N**etworks models across various types of data (vision, audio, and tabular) and with varying sample sizes. By evaluating the models on different data modalities and sample sizes, we seek to understand what factors could influence the efficiency of these two commonly used machine learning approaches.

<br>

## Introduction
---

This project compares the performance of **deep neural networks** and **decision forests** (e.g., random forests and gradient boosted trees) on structured and tabular data. While many studies have empirically compared a large number of classifiers across one or two different domains (such as 100 different tabular data settings), a careful conceptual and empirical comparison using modern best practices has yet to be conducted.

### **Conceptual Framework**
We demonstrate that both deep networks and decision forests can be viewed as "partition and vote" schemes. Specifically, both methods learn to partition the feature space into convex polytopes. For inference, each decides on the basis of votes from the activated nodes. This conceptual understanding provides a unified perspective on the relationship between these two methods.

### **Empirical Comparison**
Empirically, we compare these two strategies on several auditory and vision data settings, as well as hundreds of tabular data settings. The project involves sampling the **audio** and **vision** data according to different sample sizes, with each dataset having classification tasks for multiple classes. Additionally, for **tabular data**, we explore a diverse set of datasets for classification task, sourced from the OpenML platform. Our focus is on datasets with at most **10,000 samples**, a common size for many scientific and biomedical datasets. 

### **Findings**
In our experiments, we found that decision forests generally excel on tabular and structured data (including vision and auditory datasets) with small sample sizes. On the other hand, deep networks tend to perform better on structured data with larger sample sizes. This suggests that further improvements could be made by combining the strengths of both methods in future research.

### **Next Steps**
We will continue to refine this technical report and update the results in the coming months.

<br>

## Features
---

- Evaluate and compare the performance of deep neural networks and decision forests on various tasks
- Assess model performance across auditory, tabular, and vision datasets with varying sample sizes, as well as multi-class classification tasks for each dataset
- Comparison of raw models and tuned models for performance evaluation, analyze how tuning impacts model performance
- Explore the conceptual similarities and differences between these two modeling approaches

<br>

## Getting Started
---

The most convient way to start is clone it via Git:  
```bash
git clone https://github.com/WHHQUND/df-dn-paper.git  
```

Install the packages:  
```bash
pip install -r requirements.txt
```

<br>

## Contributing
---

<br>

## Contact
---

<br>

## License
---

<br>

## Acknowledgments
---