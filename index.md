
# Data Science: Projects and Publications

## Dissertation Research
**Using Multimodal Sentiment Analysis for Knowledge Retrieval in Structured Graphs: Understanding Social Movement Messaging from a Collection of Topical Images**

The use of computational methods on real world data to understand narrative messaging of social movements online as a way to monitor the social evolution of how a narrative can change over time, is seen across different mediums. The analysis models the communication messaging system of a social movement on a social media platform by building a narrative structure representative of the local semantics of image groups and an overall global summarization of the narrative structure as a whole. 
- *Data Collection*: Createad a customized image and metadata collector script using Selenium, Beautiful Soup and Python packaged. Collected 19,000+ images and metadata from Instagram. Cleaned and processed data with an output of ~16,000 images and metadata in the collection of topical images.
- *Data Annotation*: Used Label Studio to annotate 3,020 images based on labels identified duing visual content analysis of the images. 
- **The first step** pulls feature embedding from a collection of 16,657 Instagram images pertaining to the anti-feminicide movement in Mexico; and clusters them to identify common vector embeddings within the data. A Human-in-the-Loop (HITL) content analysis is applied to provide a qualitative understanding of the images within each cluster and to get insight on the classification aspects of the images for downstream tasks.
- **The second step** takes the images and runs them through foundational vision transformer models, along with generative Multimodal Large Language Models (M-LLMs) that generate two labels and a description of each of the images; and are compared and analyzed against each other using multi-label classification metrics along with HITL discourse analysis of the categorization process
	- *M-LLMs include:* BLIP-2, Qwen 2.5VL, Llama3, Llama4 Scout, and phi4.
- **The third step** uses the top labels created from the model with the best overall quantitative and HITL discourse evaluation metrics to build multiple graphs representing the various stages of semantics found within the image and its attributes. The graph is analyzed to identify structural patterns of the network using traditional centrality measures and modularity models (Leidan algorithm) to understand the semantic connections within and across the network. The overall graph structure is compared to benchmark datasets in a Graph Attention Network (GAT) model that uses a variational autoencoder to compare classifications of the collected annotated image data.

These quantitative findings are used in conjunction with a theoretical analysis of how the media has framed the anti-feminicide movement in Mexico since the 1990’s, and implements a review of the network’s local and global semantics creating narrative structure for the collection of topical images. 

[Dissertation Workbench: Research pipeline *in progress*](https://github.com/lwdozal/Dissertation_AI_Workbench)

[AI/ML Workshop and Codebase for M-LLMOPS Pipeline (Funded by Jetstream Grant)](https://github.com/lwdozal/AI-ML_PipelineWorkshop)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
[![](https://img.shields.io/badge/HuggingFace_Transformers-white?logo=huggingface)](#)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
[![](https://img.shields.io/badge/Instagram-white?logo=Instagram)](#) 


## Advanced Natural Language Processing Projects
**LLMs for summarization:** First project covers using Llama3.2 and Tiny Llama from the Huggingface cli to summarize an academic paper in Natural Language Processing and/or Deep Learning. The Paper summarized is "Multilingual Image Corpus – Towards a Multimodal and Multilingual Dataset" - Svetla Koeva, Ivelina Stoyanova, Jordan Kralev (2022). [Paper URL here.](https://aclanthology.org/2022.lrec-1.162/)
- [Link to project repository](https://github.com/lwdozal/ling582-fall2024-paper_summary)
  
**Multimodal & Multilingual LLMs:** The next project uses a multimodal & multiingual LLM to practice classifying images, text, and a combination of the two in a non-English language (Italian). The finetuned model used is BLIP-2, which was specifically created to provide an optimal text prompt, in different languages. With respect to this project, it helps identify the different clusters of images and image-caption pairs within a food dataset. 
- [Link to project repository](https://github.com/uazhlt-ms-program/ling-582-fall-2024-course-project-code-lwdozal_project)

**Author Profiling with Naive Bayes and Convolutional Nueral Networks:** Based on the [Wikipedia page](https://en.wikipedia.org/wiki/Author_profiling), author profiling has various techniques that can be applied to predict information about the author, which in turn can help with identifying an author from text. Here I used Naive Bayes and Convolutional Neural Networks (CNN) as my main models to provide predictions.
- [Link to project repository](https://github.com/lwdozal/ling-582-fall-2024-class-competition-code-lwdozal)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

## Laboratory of Applied Sciences - Summer Conference of Applied Data Science (LAS-SCADS)
[LAS-SCADS Website](https://ncsu-las.org/scads/) 


### Video Object Detection Project
Implemented two video object tracking models: zero-shot transformers - OWL-VIT and LMM gpt-4o on noisy data- dashcam footage of a ride in Afghanistan provided by LAS staff. The goal was to explore improvements for tracking object obfuscation and detecting unknown/unconventional objects in the noisy surveillance video data. We found that pre-trained image models provided the most accurate structured data outputs.
[Go to Project Repository](https://github.com/lwdozal/VideoObejctDetection-SCADS)

![objectDetection-Tracking](https://github.com/user-attachments/assets/596a3a63-cf61-4bc4-a536-2483fd293f60)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)

### A Comparison of Dynamic Chunk Size Methods (RAG Implememtation)
NLP researcher comparing chunk-size methods for text summarization. Used extractive summarization occams and abstractive summarization, Retrieval Augmentation Generation (RAG) to test different text segmentation methods. Used and created vector embedding database for evaluation using RAGAs, an LLM RAG evaluator. Used the 2024 TREC Challenge data-MSMarco docs and QA for analysis.
	
[Go to Project Repository](https://github.com/lwdozal/RAG_ChunkSizeS)
	
![RAGAs_eval](https://github.com/user-attachments/assets/41c39b52-d42b-4547-b7b1-135f38bb0bc4)
	
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)


## Bibliometric Data: Publishing Behavior of University of Rochester Researchers
This repository holds data and final analysis for the project, Bibliometric Data: Publishing Behavior of Rochester University Researchers. A project hosted by University of Rochester and backed by the LEADING Fellowship from Drexel University. The research question states where are University of Rochester researchers publishing/depositing data? The goal is to get a better understanding of how and where University of Rochester researchers are saving publishing their data. We attempted to analyze researcher publishing behavior and understand who is depositing data, where are they depositing it, how large are the datasets, and what formats are submitted/supported. The overall objectives were to find out where researchers are making their data publicly available. Identify common topics, relations, and overall trends. Using APIs, refine data collection techniques and conduct analyses on University of Rochester researcher data deposits into disciplinary data repositories.\
[Go to Project Repository](https://github.com/lwdozal/URoch_BibliometricBehavior)

<table>
  <tr>
    <td><img src="images/keywordTopicsPerRepository.png?raw=true"/> Kewords per Repository Topic</td>
    <td><img src="images/authors_repository_topics.png?raw=true"/>Authors Per Repository Topics</td>
    <td><img src="images/keyword_betweennessCentrality.png?raw=true"/>Keywords Betweeness Centrality</td>
  </tr>
</table>


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![RStudio](https://img.shields.io/badge/RStudio-4285F4?style=for-the-badge&logo=rstudio&logoColor=white)
[![](https://img.shields.io/badge/Anaconda-white?logo=anaconda)](#)


## Street Art Network Analysis: Applications of Bi-Partite and Bi-Dynamic Line Graphs
In this analysis, Street Art images are considered as a type of visual information that can represent a specific perception of a community as a member of a community space. Dynamic By-partite network analysis was used to understand how different neighborhoods are connected through artist attributes and how they might differ. The results show that specific neighborhood traits, urban, population, culture contribute to stronger ties within the Street Art community network.
[Street Art as Visual Information: Mixed Methods Approach to Analyzing Community Spaces](https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/pra2.537)

<img src="images/ASIS&T_LDozal_2021.jpg?raw=true"/>

[Github code: Tucson_Street-Art](/Tucson_Street-Art)

[Poster PDF](/pdf/ASIS&T_LDozal_2021.pdf)

[![](https://img.shields.io/badge/R-white?logo=R)](#)
[![](https://img.shields.io/badge/Anaconda-white?logo=anaconda)](#)
[![](https://img.shields.io/badge/Google_Maps-white?logo=Google)](#)

## Natural Language Processing: Model Comparison between BERT and Hierarchical Attention Networks
This project two natural language processing models on a dataset composed of labeled propaganda data. We reviewed off the shelf BERT and the Hierarchical Attention Network (HAN) models and found they both provide different accuracy levels, with BERT maintaining better results for the binary data classification problem.
[Identifying Propaganda: Comparing NLP Machine Learning Models on Propagandistic News Articles](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3748821)


<table>
  <tr>
    <td><img src="images/HAN.png?raw=true"/> HAN Architecture</td>
    <td><img src="images/BERT.png?raw=true"/>BERT pre-training and fine-tuning procedures</td>
  </tr>
</table>

<!-- *HAN architecture -* -->
<!-- <img src="images/HAN.png?raw=true"/> -->
[HAN Image taken from Yang et al 2016](https://aclanthology.org/N16-1174.pdf), [BERT Image taken from Delvin et al. 2019](http://arxiv.org/abs/1810.04805) 

<!-- Z. Yang, D. Yang, C. Dyer, X. He, A. Smola, and E. Hovy,“Hierarchical
Attention Networks for Document Classification,”in Proceedings of the
2016 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, San Diego,
California, 2016, pp. 14801489, doi: 10.18653/v1/N16-1174. -->

<!-- *BERT pre-training and fine-tuning procedures -* -->
<!-- <img src="images/BERT.png?raw=true"/> -->
<!-- [BERT Image taken from Delvin et al. 2019](http://arxiv.org/abs/1810.04805)  -->
<!-- J. Devlin, M. Chang, K Lee, K. Toutanova, ”BERT: Pre-training of
Deep Bidirectional Transformers for Language Understanding”, CoRR
abs/1810.04805, May 24 2019, http://arxiv.org/abs/1810.04805.-->

[Github code: Identifying Propaganda](/ECE-Final)

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) 
[![](https://img.shields.io/badge/sklearn-white?logo=scikit-learn)](#)
[![](https://img.shields.io/badge/pandas-white?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAAEDCAMAAABQ/CumAAAAeFBMVEX///8TB1QAAEb/ygDnBIgPAFLNzNYTAFnQ0NgMAFcAAETb2eP39/oUBlfV1N7/xwDmAID/9tfLydcjG17/4Yz//vbCwM3ykcL61OfoBIwyKmgAADYAAE0AAErx8PTIxdT/+un/34T85/Lyir/lAHv50eX+9fkpH2Ma8J+4AAACEklEQVR4nO3dzVIaQRSAUYNCEIGoiYmJivnP+79hFrmLVHELZ6pnmG483xqaPruh5lb32ZkkSZIkSZIkvb52z7dZU2+rT4uH2X6rx6m31afF7M1+87dTb6tPCDWEUEMINYRQQ5MS1tu0nqtMSrhKn26e1v1WmZawyn58g4DQL4QIoSyECKEshAihLIQIoSyECKEshAihLIQIoSyECKEshAihLIQIoSyECOFA6cvM5a4nYb29yjoO4WmVvM58WPQkbF8e+RqPcDlPVp4t+xLS/W0QEBCqI8yTLpsizN8n/WmJ0CEEBAQEBAQEBIT2CF+/fci6a4hw8y7rvC3CeRYCAgICAgICAgICAgICwlCEtJYIdzdp/3+kdkKHToFQ+RjJMCEcCKF7CAdC6B7CgRC6Nylh9zGtJUJ6uNCsnsOFhhkvPAHC9x+fsloi/Pp5nXTREuH++iLpMwICAgICAgICAgICAgKC/87R7/u0lggdQkBAQEBAQEB4dYQON67UTqh9KuwkDlRBQED4R8gOF5o3Rdh8yepLGO0ez6MNPO+WQ9w3NilhvBAihLIQIoSyECKEshAihLIQIoSyECKEshAihLIQIoSyECKEshAihLIQIoSyEKJt+lL0SNeADUR4TG9cGWXHew10AkPP4aRBO9ohEuOFUEMINYRQQwg1dAKEDvd41t5t2u7lL0qSJEmSJEnSyfUXeomSFq0EzbkAAAAASUVORK5CYII=)](#) [![](https://img.shields.io/badge/NumPy-white?logo=data:image/png;base64,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)](#)
[![](https://img.shields.io/badge/Spyder-white?logo=Spyder)](#)
[![](https://img.shields.io/badge/Colab-white?logo=Google)](#)
[![](https://img.shields.io/badge/Anaconda-white?logo=anaconda)](#)

<!-- [![](https://img.shields.io/badge/PyTorch-white?logo=pytorch)](#)
[![](https://img.shields.io/badge/sklearn-white?logo=scikit-learn)](#)  -->

## Mixed Method Framework for Understanding Visual Frames in Social Movements
This framework combines computational applications with visual methodologies to discover frames of meaning making in a large image collection. Frame analysis and Critical Visual Methodology (Rose, 2016) are reviewed and used in the framework to work in tangent with quantitative research methods. The methods framework is presented in the form of a matrix that enables researchers to identify applications for looking at social movements online through theoretical and computational approaches.
<table>
  <tr>
    <td><img src="images/Mixed Methods for Understanding Visual Frames in Social Movements.pptx.png?raw=true"/></td>
  </tr>
</table>

"Mixed Methods Framework for Understanding Visual Frames in Social Movements" - Long Paper Accepted to [ASIS&T 2023](https://www.asist.org/am23/2023-annual-meeting-papers/) \
Theoretical Matrix Presented at [Society for the Social Studies of Science Conference in 2022](https://4sonline.org/) <br>
[Program can be found here](https://4sonline.org/docs/Preliminary-Program-2022.pdf)

---
# Data Science: Private Datasets and Code

*Here I worked with private university data to tackle questions that support internal decision making. I used predictive modeling, statistics, Machine Learning, Data Mining, and other data analysis techniques to collect, explore, and extract insights from structured and unstructured data. Topics include revenue, retention/attrition, and student sentiment and experience.*

## Sentiment Analysis: Student Course Survey Comments

I analyzed 30K+ student course surveys using sentiment analysis packages in R to identify sentiment for all comments provided in student course surveys across campus. I also used SQL to pull the data from the Oracle database to be used for a personalized instructor dashboard.

[![](https://img.shields.io/badge/SQL-white?logo=SQL)](#) 
[![](https://img.shields.io/badge/R-white?logo=R)](#) 
[![](https://img.shields.io/badge/tidyverse-white?logo=R)](#) 
[![](https://img.shields.io/badge/dplyr-white?logo=R)](#) 
[![](https://img.shields.io/badge/Jira-white?logo=Jira)](#) 

## Survival Analysis: University Retention and Attrition

I used survival and churn analysis to analyze the expected duration of attrition for female students and The University of Arizona. Here, Kaplan Mier and the supporting cox regression analysis were used to study retention. Churn analysis including methods of logistic regression, decision trees and random forests were used to study attrition. [Read the report here](https://wise.arizona.edu/news/time-graduation-and-attrition-rates-undergraduate-women-uarizona)

[![](https://img.shields.io/badge/SQL-white?logo=SQL)](#) 
[![](https://img.shields.io/badge/R-white?logo=R)](#) 
[![](https://img.shields.io/badge/tidyverse-white?logo=R)](#) 
[![](https://img.shields.io/badge/dplyr-white?logo=R)](#) 
[![](https://img.shields.io/badge/Jira-white?logo=Jira)](#) 

## Internal Dashboard: Net Tuition Revenue

I developed an end-to-end production process of tuition and headcount dashboard visualizations and analysis using R and SQL. This consisted of aggregated data tables from an internal Oracle data warehouse that include descriptive statistics and inflation information of campus-wide Net Tuition Revenue (NTR).

[![](https://img.shields.io/badge/SQL-white?logo=SQL)](#) 
[![](https://img.shields.io/badge/Oracle-white?logo=Oracle)](#) 
[![](https://img.shields.io/badge/R-white?logo=R)](#) 
[![](https://img.shields.io/badge/Jira-white?logo=Jira)](#) 


## Campus Climate Survey: Statistical Analysis and Inference

Applied statistical analysis and inference to surveys and collected data to review and test the University of Arizona’s Medical School’s performance for future accreditation. Types of data included survey data, raw archived data, collected government and academic data from large data systems. Used both R and Python to use analysis of variance, chi-square tests, post hoc and assumptions checks, regression analysis, correlation analysis, visualizations.

[![](https://img.shields.io/badge/Python-white?logo=Python)](#)
[![](https://img.shields.io/badge/R-white?logo=R)](#) 


<!-- ---
[Project 1 Title](/sample_page)
<img src="images/dummy_thumbnail.jpg?raw=true"/>


---

### Category Name 2

- [Project 1 Title](http://example.com/)
- [Project 2 Title](http://example.com/)
- [Project 3 Title](http://example.com/)
- [Project 4 Title](http://example.com/)
- [Project 5 Title](http://example.com/)

--- -->




---
<!-- <p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p> -->
<!-- Remove above link if you don't want to attibute -->
