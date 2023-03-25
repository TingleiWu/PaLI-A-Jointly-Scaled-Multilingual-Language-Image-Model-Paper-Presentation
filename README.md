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

> Following this line of work, the paper introduced a new model called PaLI(Pathways Language and Image model). PaLI performs many image-only, language-only, and image+language tasks, across many languages, using a single “image- and-text to text” interface. A key ingredient to PaLI is the reuse of large unimodal backbones for language and vision modeling, in order to transfer existing capabilities and reduce training cost. To train PaLI, we make use of large pretrained encoder-decoder language models and Vision Transformers (ViTs). This allows us to capitalize on their existing capabilities and leverage the substantial cost of training them. We find that joint scaling of the vision and language components is important. Since existing Transformers for language are much larger than their vision counterparts, we train the largest ViT to date (ViT-e) to quantify the benefits from even larger-capacity vision models. To train PaLI, we create a large multilingual mix of pretraining tasks, based on a new image-text training set containing 10B images and texts in over 100 languages. PaLI achieves state-of- the-art in multiple vision and language tasks (such as captioning, visual question- answering, scene-text understanding), while retaining a simple, modular, and scalable design. Since existing Transformers for language are much larger than their vision counterparts, the paper also 


# Model Architecture

# Data


# Pretraining Tasks


# Testing


# Limitations/Biases


# Critical Analysis


# Link
