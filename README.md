# Alcohol Consumption GAN

This project contains simple PyTorch scripts for training a DCGAN and a small CNN classifier. The dataset consists of two folders: `dataset/sober` and `dataset/drunk`.
`augment_data.py` trains separate GANs for each class and produces extra images.

## Usage

1. Place your images in `dataset/sober` and `dataset/drunk`.
2. Run `python train_gan.py` to train a GAN on all images.
3. Execute `python augment_data.py` to create synthetic images in `augmented_dataset`.
4. Use the generated data to train the classifier in `classifier/train_classifier.py`.


