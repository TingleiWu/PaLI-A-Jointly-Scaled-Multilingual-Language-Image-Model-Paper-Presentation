# PaLI-A-Jointly-Scaled-Multilingual-Language-Image-Model-Paper-Presentation

# Outline
- [Introduction](#Introduction)
- [Ppaper Overview](# Paper Overview)
- [Model Architecture](# Model Architecture)
- [Data](# Data)
- [Pretraining Tasks](# Pretraining Tasks)
- [Testing](# Testing)
- [Limitations/Biases](# Limitations/Biases)
- [Critical Analysis](# Critical Analysis)
- [Link](# Link)

# Introduction

Increasing neural network capacity has been a successful trend in the modeling of language and vision tasks. Language models such as T5 and GPT-3 have shown significant advantages from training large Transformers on large amounts text data. On the other hand, vision models such as CNNs and Vision Transformers have seen similar benefits from scaling but to a lesser extent compared to language models. Language-and-vision modeling are also popular now dealing with problems like Image Captioning and Visual Question-Answering(VQA).

# Paper Overview

Following this line of work, the paper introduced a new model called PaLI(Pathways Language and Image model). PaLI performs many image-only, language-only, and image+language tasks, across many languages, using a single “image-and-text to text” interface. A key point to PaLI is that it reuses the large unimodal backbones for language and vision modeling, in order to transfer existing capabilities and reduce training cost. The paper combined large pretrained encoder-decoder language models and Vision Transformers (ViTs) together to train PaLI. Joint scaling of the vision and language components is an important idea in this paper. Since existing Transformers for language are much larger than their vision counterparts, this paper retrained the vision transformer and introduced the largest ViT to date (ViT-e) to quantify the benefits from even larger-capacity vision models. To train PaLI, the authors created a large multilingual mix of pretraining tasks, based on a new image-text dataset called WebLI, containing 10B images and texts in over 100 languages. Overall, PaLI generated the best result in multiple vision and language tasks (such as captioning, visual question- answering, scene-text understanding), while retaining a simple, modular, and scalable design. 


# Model Architecture

![](https://github.com/TingleiWu/PaLI-A-Jointly-Scaled-Multilingual-Language-Image-Model-Paper-Presentation/blob/main/Image_folder/LILM%20%20PaLI%2006.gif)

The model accepts image and test as the input and generates text as output. Since all tasks are performed with the same model, we use text-based prompts to indicate to the model which task to perform.
This image shows a high-level schematic of the model architecture. At its core, we have a text encoder-decoder Transformer. To include vision/image as input, the text encoder is fed with a sequence of visual “tokens”: output features of a Vision Transformer which takes as input an image. No pooling is applied to the output of the Vision Transformer before passing the visual tokens to the encoder-decoder model via cross-attention.
We reuse previously trained unimodal checkpoints. For the text encoder-decoder, we reuse pre-trained mT5 models, while for the image encoder, we reuse large vanilla ViT models.

### Visual Component

For the visual component, ViT-e has taken the place and it has the same architecture and uses the same training recipe as the 1.8B parameter ViT-G model, and it is scaled to 4B parameters. While Scaling up vision backbones leads to saturating gains on classification tasks such as ImageNet. In other words, this scailing up technique does not show much improvement in terms of image classification task. We further confirm this, observing that ViT-e is only marginally better than ViT-G on ImageNet. However, we observe substantial performance improvements from ViT-e on vision-language combined tasks in PaLI, 

### Language Component

mT5 backbone was used as our language modeling component. We used both the pretrained mT5-Large (1B parameters) and the mT5-XXL (13B parameters), from which we initialize the language encoder-decoder of PaLI. Many different tasks are trained such as pure language understanding tasks.

### Overall Model

<img width="696" alt="Screen Shot 2023-03-25 at 2 16 58 PM" src="https://user-images.githubusercontent.com/89117508/227737064-4b3f9710-0d28-4e6d-9d0f-26df62e61ea3.png">

Three models are considered:
- PaLI-3B (ViT-G: 1.8B, mT5-L: 1.2B)
- PaLI-15B (ViT-G: 1.8B, mT5-XXL: 13B)
- PaLI-17B (ViT-e: 3.9B, mT5-XXL: 13B)


# Data


# Pretraining Tasks


# Testing


# Limitations/Biases


# Critical Analysis


# Link
