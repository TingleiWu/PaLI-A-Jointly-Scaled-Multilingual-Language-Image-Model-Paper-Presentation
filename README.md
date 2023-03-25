# PaLI-A-Jointly-Scaled-Multilingual-Language-Image-Model-Paper-Presentation

# Outline
- [Introduction](#introduction)
- [Paper Overview](#paper-overview)
- [Model Architecture](#model-architecture)
- [Data](#data)
- [Pretraining Tasks](#pretraining-tasks)
- [Testing](#testing)
- [Limitations/Biases](#limitations/Biases)
- [Critical Analysis](#critical-analysis)
- [Link](#link)

# Introduction

Increasing neural network capacity has been a successful trend in the modeling of language and vision tasks. Language models such as T5 and GPT-3 have shown significant advantages from training large Transformers on large amounts text data. On the other hand, vision models such as CNNs and Vision Transformers have seen similar benefits from scaling but to a lesser extent compared to language models. Language-and-vision modeling are also popular now dealing with problems like Image Captioning and Visual Question-Answering(VQA).

# Paper Overview

Following this line of work, the paper introduced a new model called PaLI(Pathways Language and Image model). PaLI performs many image-only, language-only, and image+language tasks, across many languages, using a single “image-and-text to text” interface. A key point to PaLI is that it reuses the large unimodal backbones for language and vision modeling, in order to transfer existing capabilities and reduce training cost. The paper combined large pretrained encoder-decoder language models and Vision Transformers (ViTs) together to train PaLI. Joint scaling of the vision and language components is an important idea in this paper. Since existing Transformers for language are much larger than their vision counterparts, this paper retrained the vision transformer and introduced the largest ViT to date (ViT-e) to quantify the benefits from even larger-capacity vision models. To train PaLI, the authors created a large multilingual mix of pretraining tasks, based on a new image-text dataset called WebLI, containing 10B images and texts in over 100 languages. Overall, PaLI generated the best result in multiple vision and language tasks (such as captioning, visual question- answering, scene-text understanding), while retaining a simple, modular, and scalable design. 


# Model Architecture

![](https://github.com/TingleiWu/PaLI-A-Jointly-Scaled-Multilingual-Language-Image-Model-Paper-Presentation/blob/main/Image_folder/LILM%20%20PaLI%2006.gif)

The model accepts image and text as the input and generates text as output. Since all tasks are performed with the same model, we use text-based prompts to indicate to the model which task to perform.
This image shows a high-level schematic of the model architecture. At its core, we have a text encoder-decoder Transformer. To include vision/image as input, the text encoder is fed with a sequence of visual “tokens”: output features of a Vision Transformer which takes as input an image. No pooling is applied to the output of the Vision Transformer before passing the visual tokens to the encoder-decoder model via cross-attention. We reuse previously trained unimodal checkpoints. For the text encoder-decoder, we reuse pre-trained mT5 models, while for the image encoder, we reuse large vanilla ViT models.

### Visual Component

<img width="409" alt="Screen Shot 2023-03-25 at 4 15 26 PM" src="https://user-images.githubusercontent.com/89117508/227742456-5bd70bbf-60d5-43fb-9ba4-f91a168c2df4.png">

For the visual component, ViT-e has taken the place and it has the same architecture and uses the same training recipe as the 1.8B parameter ViT-G model, and it is scaled to 4B parameters. Scaling up vision backbones leads to saturating gains on classification tasks such as ImageNet. In other words, this scailing up technique does not show much improvement in terms of image classification task. We further confirm this, observing that ViT-e is only marginally better than ViT-G on ImageNet. However, we observe substantial performance improvements from ViT-e on vision-language combined tasks in PaLI, 

### Language Component

mT5 backbone was used as our language modeling component. We used both the pretrained mT5-Large (1B parameters) and the mT5-XXL (13B parameters), from which we initialize the language encoder-decoder of PaLI. Many different tasks are trained such as pure language understanding tasks.

### Overall Model

<img width="696" alt="Screen Shot 2023-03-25 at 2 16 58 PM" src="https://user-images.githubusercontent.com/89117508/227737064-4b3f9710-0d28-4e6d-9d0f-26df62e61ea3.png">

Three models are considered:
- PaLI-3B (ViT-G: 1.8B, mT5-L: 1.2B)
- PaLI-15B (ViT-G: 1.8B, mT5-XXL: 13B)
- PaLI-17B (ViT-e: 3.9B, mT5-XXL: 13B)


# Data
<img width="749" alt="Screen Shot 2023-03-25 at 3 05 31 PM" src="https://user-images.githubusercontent.com/89117508/227739358-719502c3-e726-490f-b297-1e729c89db10.png">

<img width="718" alt="Screen Shot 2023-03-25 at 3 14 01 PM" src="https://user-images.githubusercontent.com/89117508/227739411-aae6b871-4b2c-47bc-abdd-20965d51d1a9.png">

WebLI, a multilingual image-language dataset built from images and texts available on the public web, was used to train the PaLI model. WebLI scales up the image language data collection from English-only datasets to 109 languages, which enables us to pretrain PaLI multilingually, and perform downstream tasks across many languages. Due to the abundance of multilingual content on the internet, the collection process for the WebLI dataset can be scaled to cover 10 billion images and 12 billion alt-texts. In addition to annotation with web text, we apply the GCP Vision API to extract OCR annotations on all images, resulting in 29 billion image-OCR pairs.

De-duplication applied on this WebLI dataset to mitigate train-to-test leakage. This was done by removing images against 68 common vision/vision-language datasets. Eliminating these images from the WebLI dataset does not result in any significant shrinkage (0.36%).


# Pretraining Tasks

To accommodate diverse tasks in the image-language space, we train PaLI using a mixture of pre-training tasks:
- Span corruption on text-only data
- Split-captioning (SplitCap) on WebLI alt-text data
- OCR on WebLI OCR-text data
- English and Cross-Lingual VQA
- English-only Object-Aware (OA) VQA
- Object detection

The overall size of the data we use for pretraining is 1.6B examples. This dataset is comparable, but slightly smaller and designed to be cleaner than the datasets used in SimVLM (1.8B), CoCa (1.8B), and Flamingo (2.3B). However, unlike other datasets mentioned above, WebLI is multilingual, so the 1.6B examples follow a long-tailed distribution over the 100+ languages covered.

### Training detail for the model

For the learning rate, we use a 1k-step linear warmup, followed by inverse square-root decay. For PaLI-3B, we use a peak learning rate of 0.01. For larger models, PaLI-15B and PaLI-17B, we use a peak learning rate of 0.005. The largest model, PaLI-17B, is pretrained using 1,024 GCP-TPUv4 chips for 7 days. It uses a four-way model partitioning and a batch size of 4,096. Overall, the model passes over 1.6B images, one epoch over the entire pretraining dataset. The image resolution for this pass is 224×224, and they later changed the image resolution to be 588×588 for pre-finetuning. During training, only the parameters of the language component are updated and the vision component is frozen, which provides a boost in performance.


# Testing

We evaluate on multiple downstream tasks that include a number of vision and language benchmarks, and additionally language-only and vision-only benchmarks.

### Image Captioning(COCO Captions, NoCaps, TextCaps, Multilingual captioning on XM-3600)

<img width="514" alt="Screen Shot 2023-03-25 at 4 48 45 PM" src="https://user-images.githubusercontent.com/89117508/227744253-54f3094f-2dcb-45df-b95d-a1eb6a80efa2.png">

PaLI outperformed all the other models and established a new high at 149.1 CIDEr(Consensus-based Image Description Evaluation) points for COCO Captions. PaLI generated similar but suboptimal results for NoCaps compare to GIT2, and it showed subtantial performance improvement for TextCaps.

<img width="706" alt="Screen Shot 2023-03-25 at 5 01 09 PM" src="https://user-images.githubusercontent.com/89117508/227744396-6d9c98f5-2054-436e-b96e-bad161ee7e9d.png">

CIDEr scores on image captioning for the Crossmodal-3600 benchmark, covering seven diverse languages (English, French, Hindi, Hebrew, Romanian, Thai, and Chinese), as well as the average of the 35 languages covered by the benchmark

### Visual Question Answering(VQAv2, OKVQA, TextVQA, VizWiz-QA)

all of the VQA results reported in this paper are performed in the open-vocabulary setting using the 250k mT5 vocabulary. Most prior works use the VQA-as-classification setting, where a best answer among a predefined set (usually of size 3k) needs to be selected. Note that the VQA-as-open-generation setting is challenging because: (1) The generated text is directly compared to the desired answer and only an exact match is counted as accurate. (2) The PaLI vocabulary covers 100+ languages and is significantly larger than both those used in the classification setting, and those used by previous single-language open-generation models 

<img width="654" alt="Screen Shot 2023-03-25 at 5 17 34 PM" src="https://user-images.githubusercontent.com/89117508/227744961-637b693e-7e90-423d-a421-ffd06fedabfb.png">

VQA Accuracy results on VQAv2, OKVQA, TextVQA, and VizWiz-QA. PaLI models are evaluated in the open-vocabulary generation setting, and still outperform previous models that use closed-vocabulary classification evaluations (SimVLM, CoCa, BEiT3, OFA). Mia (with “†”) is the winning model of TextVQA Challenge 2021. Numbers shown in gray are from models using closed-vocabulary classification. OKVQA is the benchmark that requires external knowledge to answer its questions, that is, knowledge that is not directly present in the image input, and instead needs to be indirectly inferred by the model. Therefore, the results from Flamingo and PaLI-17B suggest that leveraging external knowledge does not necessarily require specific training, and instead can be achieved with generic large-capacity models trained on large amounts of data.






# Limitations/Biases


# Critical Analysis


# Link

- [Introduction to CIDEr](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf)
- [Webli Data Card](https://github.com/google-research/google-research/blob/master/pali/webli_data_card.pdf)
- [PaLI Model Card](https://github.com/google-research/google-research/blob/master/pali/pali_model_card.pdf)
- [Vison Transformer](https://viso.ai/deep-learning/vision-transformer-vit/#:~:text=The%20ViT%20is%20a%20visual,class%20labels%20for%20the%20image.)
