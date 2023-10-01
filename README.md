# Estimating Uncertainty in Generative Adversarial Neural Networks for Super-Resolution

## Description
This repository contains the implementation and supplementary materials for the Master's Thesis titled "Estimating Uncertainty in Generative Adversarial Neural Networks for Super-Resolution" by Maniraj Sai Adapa, submitted to the University of Groningen.

## Brief Summary
This thesis explores the integration of uncertainty estimation techniques with Generative Adversarial Neural Networks (GANs), specifically focusing on Super-Resolution GAN (SRGAN) and Enhanced Super-Resolution GAN (ESRGAN). The primary objective is to enhance the reliability and interpretability of these models by quantifying the uncertainty in their outputs, addressing the limitations posed by noise and artifacts in image super-resolution.

## Purpose
The purpose of this research is to advance super-resolution capabilities through uncertainty modeling, thereby laying the groundwork for safer and more reliable utilization of AI in various domains, including healthcare, security, and surveillance. By combining industry-standard GANs with uncertainty estimation techniques like Monte Carlo dropout and ensemble methods, the study aims to provide a measure of the reliability of the reconstructed image, allowing for improvements in the reconstruction process and enabling well-informed decision-making in critical applications.

## Main Findings
1. ### Uncertainty-Aware Super-Resolution:
   - The integration of uncertainty estimation with super-resolution GANs produces enhanced images with quantified reliability, enabling the pinpointing of regions in the image where confidence is lower.
   - This approach aids in distinguishing between genuine anomalies and artifacts produced by the model, supporting precise interventions and well-informed decisions in critical domains like medical imaging and aerial imaging.

2. ### Optimal Method for Uncertainty Estimation:
   - The study compares different uncertainty estimation techniques to determine the optimal method in terms of performance and impact on the stability of the training process and convergence properties of GANs.

3. ### Enhanced Interpretability and Reliability:
   - The incorporation of uncertainty modeling enhances the interpretability and reliability of deep learning algorithms, providing insights into the confidence or certainty associated with each prediction and allowing researchers and professionals to assess the potential risks associated with deploying the algorithms.
