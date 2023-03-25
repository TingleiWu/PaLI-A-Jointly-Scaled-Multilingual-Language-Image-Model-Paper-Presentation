# PaLI-A-Jointly-Scaled-Multilingual-Language-Image-Model-Paper-Presentation

# Outline
- [Introduction](#Overview)
- [Paper Overview](#Paper Overview)
- [Model Architecture](#Model Architecture)
- [Data](#Data)
- [Pretraining Tasks](#Pretraining Tasks)
- [Testing](#Testing)
- [Limitations/Biases](#Limitations/Biases)
- [Critical Analysis](#Critical Analysis)
- [Link](#Link)

# Overview

> Increasing neural network capacity has been a successful trend in the modeling of language and vision tasks. Language models such as T5 and GPT-3 have shown significant advantages from training large Transformers on large amounts text data. On the other hand, vision models such as CNNs and Vision Transformers have seen similar benefits from scaling but to a lesser extent compared to language models. Language-and-vision modeling are also popular now dealing with problems like Image Captioning and Visual Question-Answering.

# Paper Overview

> Following this line of work, the paper introduced a new model called PaLI(Pathways Language and Image model). PaLI performs many image-only, language-only, and image+language tasks, across many languages, using a single “image-and-text to text” interface. A key point to PaLI is that it reuses the large unimodal backbones for language and vision modeling, in order to transfer existing capabilities and reduce training cost. The paper combined large pretrained encoder-decoder language models and Vision Transformers (ViTs) together to train PaLI. This allows the model to capitalize on the existing capabilities and reduce the cost of training. Joint scaling of the vision and language components is an important idea in this paper. Since existing Transformers for language are much larger than their vision counterparts, this paper retrained the vision transformer and introduced the largest ViT to date (ViT-e) to quantify the benefits from even larger-capacity vision models. To train PaLI, The authors created a large multilingual mix of pretraining tasks, based on a new image-text dataset called WebLI, containing 10B images and texts in over 100 languages. PaLI generated the best result in multiple vision and language tasks (such as captioning, visual question- answering, scene-text understanding), while retaining a simple, modular, and scalable design. 


# Model Architecture

# Data


# Pretraining Tasks


# Testing


# Limitations/Biases


# Critical Analysis


# Link
