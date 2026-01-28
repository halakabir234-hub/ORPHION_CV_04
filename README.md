#ORPHION_CV_04
#Image-to-Image Translation using Pix2Pix (Conditional GAN)
#📘 Context

Image-to-image translation is a fundamental problem in computer vision where the goal is to learn a mapping between two visual domains. Instead of generating images from random noise, conditional Generative Adversarial Networks (cGANs) learn to transform one image into another while preserving structural information.

This project implements Pix2Pix, a popular cGAN architecture introduced in the paper
“Image-to-Image Translation with Conditional Adversarial Networks” (Isola et al., 2017).

Pix2Pix has been successfully applied to tasks such as:

Converting architectural labels into realistic building facades

Colorizing black-and-white images

Transforming sketches into photographs

Aerial-to-map image translation

This repository contains Task-04 of the ORPHION Internship Program, focusing on Computer Vision (CV) using TensorFlow.

📌 Project Overview

In this task, a Pix2Pix conditional GAN is trained to generate realistic building facade images from architecture label images using the CMP Facade Dataset.

The model consists of:

Generator: U-Net–based architecture with skip connections

Discriminator: PatchGAN classifier that evaluates image realism at the patch level

The implementation closely follows the official TensorFlow Pix2Pix tutorial, with structured preprocessing, training, and visualization steps.

🧠 Architecture Summary
Generator (U-Net)

Encoder: Convolution → BatchNorm → LeakyReLU

Decoder: Transposed Convolution → BatchNorm → (Dropout) → ReLU

Skip connections between encoder and decoder layers

Discriminator (PatchGAN)

Classifies whether image patches are real or fake instead of the whole image

Encourages sharper and more realistic outputs

📂 Dataset

CMP Facade Database

Each image has size 256 × 512

Left half: Real building facade

Right half: Architecture label image

The dataset is automatically downloaded inside Google Colab.

dataset_name = "facades"
_URL = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz"

🔄 Data Preprocessing

The following preprocessing steps are applied:

Resize images to 286 × 286

Random crop back to 256 × 256

Random horizontal flipping

Normalize pixel values to [-1, 1]

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image

🧪 Training Details

Framework: TensorFlow

Batch Size: 1 (as recommended in the Pix2Pix paper)

Optimizer: Adam

Loss Functions:

Adversarial loss

L1 loss for pixel-level similarity

Hardware: Google Colab GPU (V100 recommended)

Training Time: ~15 seconds per epoch

🖼️ Sample Output

After training, the model successfully translates architecture label images into realistic building facades while preserving spatial structure.

(Add screenshots or demo video here — muted video is acceptable as per submission rules.)

▶️ How to Run the Project (Google Colab)
Step 1: Download the Notebook

Download the project notebook from this repository

The file is provided in .ipynb format

Step 2: Open Google Colab

Go to https://colab.research.google.com

Click File → Upload notebook

Upload the downloaded .ipynb file

Step 3: Enable GPU

Click Runtime → Change runtime type

Select:

Hardware accelerator: GPU

Click Save

Step 4: Run the Notebook

Run all cells sequentially from top to bottom

The dataset will download automatically

Training and visualization will execute inside Colab

✅ No local setup required
✅ No manual dataset download needed

📁 Repository Structure
ORPHION_CV_04/
│
├── ORPHION_Image_Translation_Pix2Pix_Task_04.ipynb
├── README.md

📚 Reference

TensorFlow Pix2Pix Tutorial
pix2pix: Image-to-image translation with a conditional GAN

Isola et al., 2017 — Image-to-Image Translation with Conditional Adversarial Networks

CMP Facade Dataset, Czech Technical University in Prague

✨ Key Takeaways

Learned how conditional GANs differ from standard GANs

Implemented Pix2Pix using TensorFlow from scratch

Understood U-Net architectures and PatchGAN discriminators

Gained hands-on experience with image preprocessing pipelines

Explored real-world image translation tasks in computer vision
