# Bibliography

This file tracks all research papers studied for the project:

**Multi-Class Herbal Plant Leaf (Basil, Aloe Vera, Mint) Disease Detection and Classification using YOLO26 Architecture**

---

## Paper 1

**Title:** An Open Access Repository of Images on Plant Health to Enable the Development of Mobile Disease Diagnostics

**Authors:** David P. Hughes, Marcel Salathé

**Year:** 2015

**Journal:** Scientific Data (Nature Portfolio)

**DOI:** 10.1038/sdata.2015.41

**Dataset:** PlantVillage

**Status:** ✅ Completed

**Importance:** ⭐⭐⭐⭐⭐

**Why I Read This Paper**
To understand the PlantVillage dataset, how it was collected, why it became the standard benchmark for plant disease detection, and what lessons can be applied to my own herbal plant disease dataset.

**Main Contribution**
Introduced the PlantVillage dataset containing 54,309 expert-labeled images of healthy and diseased crop leaves, enabling large-scale AI research in plant disease detection.

**How It Influences My Project**
I will use PlantVillage as one of my benchmark datasets but will also collect real smartphone images of basil, aloe vera, and mint leaves to improve model robustness in real-world conditions.

**Notes**
papers/notes/01_PlantVillage.md

---

## Paper 2

**Title:** Uncovering Bias in the PlantVillage Dataset

**Author:** Mehmet Alican Noyan

**Year:** 2022

**Source:** arXiv

**Status:** ✅ Completed

**Importance:** ⭐⭐⭐⭐⭐

**Why I Read This Paper**

To understand why models trained on the PlantVillage dataset often achieve very high accuracy but fail on real-world images, and to learn how dataset bias affects plant disease detection systems.

**Main Contribution**

Demonstrated that PlantVillage contains significant capture bias. A model trained using only 8 background pixels achieved 49% accuracy, proving that hidden dataset characteristics can influence predictions instead of actual disease symptoms.

**How It Influences My Project**

This paper motivates me to collect diverse smartphone images of basil, aloe vera, and mint leaves under different backgrounds, lighting conditions, and environments. It also reinforces the importance of evaluating the model on unseen real-world images rather than relying solely on benchmark dataset performance.

**Notes**

papers/notes/02_PlantVillage_Bias.md

---

---

## Paper 03

**Title:** A Systematic Literature Review on Deep Learning-Based Plant Leaf Disease Detection using Vision Transformers and CNNs

**Type:** Survey / Literature Review

**Year:** 2023

**Why I Read It**

To understand the evolution of deep learning methods used in plant disease detection, compare CNNs, Vision Transformers, YOLO, GANs, and hybrid models, and identify current research gaps relevant to my herbal plant disease detection project.

**Key Contribution**

Provides a comprehensive comparison of state-of-the-art deep learning techniques for plant disease detection, summarizes commonly used datasets, discusses strengths and weaknesses of different architectures, and highlights future research directions.

**Relevance to My Project**

★★★★★ (5/5)

Useful for:
- Literature Review
- Model Selection
- Understanding CNN vs ViT vs YOLO
- Identifying research gaps
- Choosing evaluation metrics
- Justifying the use of YOLO26

**Main Lesson**

Modern plant disease detection is shifting from traditional CNNs toward Vision Transformers, hybrid architectures, and real-time YOLO-based detection, while future work should focus on robust real-world datasets and deployment-ready models.
-----
## Paper 04

**Title:** Plant Leaf Disease Detection: A Comprehensive Survey with Explainable AI (XAI)

**Authors:** Computer Vision and Biometrics Lab, IIIT Allahabad; University at Buffalo

**Year:** 2024

**Type:** Survey / Review Paper

**Topics:**
- Plant Disease Detection
- CNN
- Vision Transformers
- Explainable AI (XAI)
- PlantVillage Dataset
- Plant Pathology Dataset
- Deep Learning in Agriculture

**Key Contributions:**
- Reviews traditional ML, CNNs, and Vision Transformers for plant disease detection.
- Summarizes major public datasets used in agricultural AI.
- Highlights the importance of Explainable AI (XAI) for building trustworthy disease diagnosis systems.
- Identifies future research directions such as field-ready datasets, lightweight models, and interpretable AI.

**Relevance to KrishiAI:** ⭐⭐⭐⭐⭐ (5/5)

**Reason:** Strong reference for the literature review, dataset comparison, model selection, and the future XAI component of the KrishiAI project.

-----

## Paper 05

**Title:** Leveraging YOLO Deep Learning Models to Enhance Plant Disease Identification

**Authors:** Yousef Alhwaiti, Muntazir Khan, et al.

**Journal:** Scientific Reports (Nature Portfolio)

**Year:** 2025

**Type:** Research Paper

**Topics:**
- YOLOv3
- YOLOv4
- Object Detection
- Plant Disease Detection
- Precision Agriculture
- PlantVillage Dataset
- Real-Time Computer Vision

**Key Contributions:**
- Applies YOLOv3 and YOLOv4 for automated plant disease detection and localization.
- Demonstrates that YOLOv4 achieves higher accuracy (98%), higher mAP (98%), and much faster inference than YOLOv3.
- Uses PlantVillage images of peach and strawberry leaves to evaluate disease detection performance.
- Highlights the advantages of real-time object detection over traditional image classification for agricultural applications.

**Relevance to KrishiAI:** ⭐⭐⭐⭐⭐ (5/5)

**Reason:** This paper provides strong justification for selecting YOLO-based object detection in KrishiAI. It supports the choice of YOLO architectures for real-time disease localization, benchmark comparison, and performance evaluation, forming an excellent bridge between traditional CNN-based approaches and newer models such as YOLO26.

-----

## Paper 06

**Title:** Detection and Identification of Plant Leaf Diseases using YOLOv4

**Authors:** Eman Abdullah Aldakheel, Mohammed Zakariah, et al.

**Journal:** Frontiers in Plant Science

**Year:** 2024

**Type:** Research Paper

**Topics:**
- YOLOv4
- Object Detection
- Plant Disease Detection
- PlantVillage Dataset
- Data Augmentation
- Bounding Box Localization
- Real-Time Detection
- Agricultural AI

**Key Contributions:**
- Developed a YOLOv4-based framework for simultaneous plant disease detection and localization.
- Achieved very high performance on the PlantVillage dataset using optimized YOLOv4 training.
- Demonstrated the importance of annotation quality, data augmentation, and hyperparameter tuning for agricultural object detection.
- Compared YOLOv4 with ResNet, DenseNet, YOLOv5, CNN, and SVM, showing superior performance in their experiments.
- Highlighted future directions including larger field datasets, lightweight models, and more robust real-world deployment.

**Relevance to KrishiAI:** ⭐⭐⭐⭐⭐ (5/5)

**Reason:** One of the most relevant papers for KrishiAI because it directly applies YOLO-based object detection to plant disease diagnosis. It provides guidance on annotation, training, augmentation, evaluation, and deployment strategies that can be adopted for a YOLO26-based disease detection system.

-------

## Paper 07

**Title:** Plant Disease Classification of Basil and Mint Leaves using Convolutional Neural Networks

**Authors:** V. Sathiya, M. S. Josephine, V. Jeyabalaraja

**Journal:** International Journal of Intelligent Systems and Applications in Engineering (IJISAE)

**Year:** 2023

**Type:** Research Paper

**Topics:**
- Basil leaf disease classification
- Mint leaf disease classification
- CNN
- InceptionV3
- Herbal plant disease detection
- Image augmentation
- Plant leaf preprocessing

**Key Contributions:**
- Proposes a CNN-based system for basil and mint leaf disease classification.
- Uses InceptionV3 as the best-performing architecture.
- Builds a multi-class herbal plant dataset with healthy and diseased basil and mint classes.
- Applies preprocessing and augmentation to improve classification performance.
- Reports validation accuracy of 77.55% for basil and 70.89% for mint.

**Relevance to KrishiAI:** ⭐⭐⭐⭐☆ (4/5)

**Reason:** Useful as a baseline paper for basil and mint disease classification and for dataset design ideas, but it is classification-focused rather than detection-focused. It is still valuable for the literature review because it supports the herbal-plant angle of my project.

------

