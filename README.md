# PaLI-A-Jointly-Scaled-Multilingual-Language-Image-Model-Paper-Presentation

# Outline
- [Introduction](#introduction)
- [Paper Overview](#paper-overview)
- [Model Architecture](#model-architecture)
- [Data](#data)
- [Pretraining Tasks](#pretraining-tasks)
- [Testing](#testing)
- [Critical Analysis](#critical-analysis)
- [Questions](#questions)
- [Link](#link)

# Introduction

Increasing neural network capacity has been a successful trend in the modeling of language and vision tasks. Language models such as T5 and GPT-3 have shown significant advantages from training large Transformers on large amounts text data. On the other hand, vision models such as CNNs and Vision Transformers have seen similar benefits from scaling but to a lesser extent compared to language models. Language-and-vision models like COCA and Florence are also popular now dealing with problems like Image Captioning and Visual Question-Answering(VQA).

# Paper Overview

Following this line of work, the paper introduced a new model called PaLI(Pathways Language and Image model). PaLI performs many image-only, language-only, and image+language tasks, across many languages, using a single “image-and-text to text” interface. A key point to PaLI is that it reuses the large unimodal backbones for language and vision modeling, in order to transfer existing capabilities and reduce training cost. The paper combined large pretrained encoder-decoder language models and Vision Transformers (ViTs) together to train PaLI. Joint scaling of the vision and language components is an important idea in this paper. Since existing Transformers for language are much larger than their vision counterparts, this paper retrained the vision transformer and introduced the largest ViT to date (ViT-e). To train PaLI, the authors created a large multilingual mix of pretraining tasks, based on a new image-text dataset called WebLI, containing 10B images and texts in over 100 languages. Overall, PaLI generated the best result in multiple vision and language tasks such as captioning, visual question-answering, scene-text understanding, while retaining a simple, modular, and scalable design. 


# Model Architecture

![](https://github.com/TingleiWu/PaLI-A-Jointly-Scaled-Multilingual-Language-Image-Model-Paper-Presentation/blob/main/Image_folder/LILM%20%20PaLI%2006.gif)

The model accepts image and text as the input and generates text as output. Becaseu this model can perform different tasks, text-based prompt is used to indicate to the model which task to perform.
At its core, we have a text encoder-decoder Transformer. To include vision/image as input, the text encoder is fed with a sequence of visual “tokens”: output features of a Vision Transformer which takes as input an image. No pooling is applied to the output of the Vision Transformer before passing the visual tokens to the encoder-decoder model. The authors reused previously trained unimodal as the starting point. 

### Visual Component

<img width="409" alt="Screen Shot 2023-03-25 at 4 15 26 PM" src="https://user-images.githubusercontent.com/89117508/227742456-5bd70bbf-60d5-43fb-9ba4-f91a168c2df4.png">

For the visual component, ViT-e has taken the place and it has the same architecture and uses the same training recipe as the 1.8B parameter ViT-G model, and it is scaled to 4B parameters. Scaling up vision backbones leads to saturating gains on classification tasks such as ImageNet. In other words, this scailing up technique does not show much improvement in terms of image classification task. However, substantial performance improvements observed from ViT-e on vision-language combined tasks in PaLI, 

### Language Component

mT5 backbone was used as our language modeling component. Both the pretrained mT5-Large (1B parameters) and the mT5-XXL (13B parameters) were used, from which we initialize the language encoder-decoder of PaLI. Many different tasks are trained such as pure language understanding tasks.

### Overall Model

<img width="696" alt="Screen Shot 2023-03-25 at 2 16 58 PM" src="https://user-images.githubusercontent.com/89117508/227737064-4b3f9710-0d28-4e6d-9d0f-26df62e61ea3.png">

Three models are considered:
- PaLI-3B (ViT-G: 1.8B, mT5-L: 1.2B)
- PaLI-15B (ViT-G: 1.8B, mT5-XXL: 13B)
- PaLI-17B (ViT-e: 3.9B, mT5-XXL: 13B)


# Data
<img width="749" alt="Screen Shot 2023-03-25 at 3 05 31 PM" src="https://user-images.githubusercontent.com/89117508/227739358-719502c3-e726-490f-b297-1e729c89db10.png">

<img width="718" alt="Screen Shot 2023-03-25 at 3 14 01 PM" src="https://user-images.githubusercontent.com/89117508/227739411-aae6b871-4b2c-47bc-abdd-20965d51d1a9.png">

WebLI, a multilingual image-language dataset built from images and texts available on the public web, was used to train the PaLI model. WebLI scales up the image language data collection from English-only datasets to 109 languages, which enables us to pretrain PaLI multilingually, and perform downstream tasks across many languages. Due to the abundance of multilingual content on the internet, the collection process for the WebLI dataset can be scaled to cover 10 billion images and 12 billion alt-texts. In addition to annotation with web text, the paper extracted OCR annotations on all images, resulting in 29 billion image-OCR pairs.

De-duplication applied on this WebLI dataset to mitigate train-to-test leakage. This was done by removing images against 68 common vision/vision-language datasets. Eliminating these images from the WebLI dataset does not result in any significant shrinkage (0.36%).


# Pretraining Tasks

To accommodate diverse tasks in the image-language space, we train PaLI using a mixture of pre-training tasks:
- Span corruption on text-only data
- Split-captioning (SplitCap) on WebLI alt-text data
- OCR on WebLI OCR-text data
- English and Cross-Lingual VQA
- English-only Object-Aware (OA) VQA
- Object detection

The overall size of the data we use for pretraining is 1.6B examples. This dataset is comparable, but slightly smaller and designed to be cleaner than the datasets used in SimVLM (1.8B), CoCa (1.8B), and Flamingo (2.3B). However, unlike other datasets, WebLI is multilingual, so the 1.6B examples follow a long-tailed distribution over the 100+ languages covered.

### Training detail for the model

For the learning rate, the authors used a 1k-step linear warmup, followed by inverse square-root decay. For PaLI-3B, they use a peak learning rate of 0.01. For larger models, PaLI-15B and PaLI-17B, they use a peak learning rate of 0.005. Overall, the model passes over 1.6B images, one epoch over the entire pretraining dataset. The image resolution for this pass is 224×224, and they later changed the image resolution to be 588×588 for pre-finetuning. 


# Testing

### Image Captioning(COCO Captions, NoCaps, TextCaps, Multilingual captioning on XM-3600)

<img width="514" alt="Screen Shot 2023-03-25 at 4 48 45 PM" src="https://user-images.githubusercontent.com/89117508/227744253-54f3094f-2dcb-45df-b95d-a1eb6a80efa2.png">

PaLI outperformed all the other models and established a new high at 149.1 CIDEr(Consensus-based Image Description Evaluation) points for COCO Captions. PaLI generated similar but suboptimal results for NoCaps compare to GIT2, and it showed subtantial performance improvement for TextCaps.

<img width="706" alt="Screen Shot 2023-03-25 at 5 01 09 PM" src="https://user-images.githubusercontent.com/89117508/227744396-6d9c98f5-2054-436e-b96e-bad161ee7e9d.png">

CIDEr scores on image captioning for the Crossmodal-3600 benchmark(multilingual), covering seven diverse languages (English, French, Hindi, Hebrew, Romanian, Thai, and Chinese), as well as the average of the 35 languages covered by the benchmark

### Visual Question Answering(VQAv2, OKVQA, TextVQA, VizWiz-QA)

All of the VQA results reported in this paper are performed in the open-vocabulary setting. Most prior works use the VQA-as-classification setting, where a best answer among a predefined set (usually of size 3k) needs to be selected. Note that the VQA-as-open-generation setting is challenging because the generated text is directly compared to the desired answer and only an exact match is counted as accurate. 

<img width="654" alt="Screen Shot 2023-03-25 at 5 17 34 PM" src="https://user-images.githubusercontent.com/89117508/227744961-637b693e-7e90-423d-a421-ffd06fedabfb.png">

PaLI models are evaluated in the open-vocabulary generation setting, and still outperform previous models that use closed-vocabulary classification evaluations (SimVLM, CoCa, BEiT3, OFA). Numbers shown in gray are from models using closed-vocabulary classification. Mia (with “†”) is the winning model of TextVQA Challenge 2021. OKVQA is the benchmark that requires external knowledge to answer its questions, that is, knowledge that is not directly present in the image input, and instead needs to be indirectly inferred by the model. Therefore, the results from Flamingo and PaLI-17B suggest that leveraging external knowledge does not necessarily require specific training, and instead can be achieved with generic large-capacity models trained on large amounts of data.

### Language-understanding Capabilities

Since PaLI is pretrained with a diverse mixture of multimodal tasks with image and text data, it raises the question on whether it would “forget” its language modeling capability. Therefore, we compare mT5-XXL and PaLI-17B on a range of language understanding benchmarks, including the English-only SuperGLUE benchmark and three multilingual benchmarks: XNLI, XQuAD, TyDiQA-GoldP. 

<img width="666" alt="Screen Shot 2023-03-25 at 5 30 14 PM" src="https://user-images.githubusercontent.com/89117508/227745380-527df346-1d41-41db-a4cb-f715853d455f.png">

The first row is the result reported by its original paper. The second row is the result using the publicly available mT5-XXL checkpoint, which is also the starting point for PaLI-17B. The third row results are using the trained PaLI-17B model.


# Critical Analysis
- The paper showed some limitations and potential drawbacks of the proposed model, but it failed to consider them in detail. For example, it is unclear how well the model generalizes to out-of-distribution languages or how sensitive it is to the quality and size of the image and text data.
- The paper does not provide a detailed analysis of the computational complexity and scalability of the proposed model. Nowadays, it is important to evaluate and understand the practical feasibility of training and deploying such models in real-world scenarios.

# Questions
- What are the potential applications of PaLI in industry or real-world scenarios that you can think of?
- What ethical considerations should be taken into account when building and deploying multilingual language-image models like PaLI, particularly with regards to issues of bias, representation, and cultural sensitivity?



# Link

- [Introduction to CIDEr](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf)
- [Webli Data Card](https://github.com/google-research/google-research/blob/master/pali/webli_data_card.pdf)
- [PaLI Model Card](https://github.com/google-research/google-research/blob/master/pali/pali_model_card.pdf)
- [Vison Transformer](https://viso.ai/deep-learning/vision-transformer-vit/#:~:text=The%20ViT%20is%20a%20visual,class%20labels%20for%20the%20image.)
- [Multilingual captioning on XM-3600](https://arxiv.org/pdf/2205.12522.pdf)
