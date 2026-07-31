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