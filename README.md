# Alcohol Consumption GAN

This project contains simple PyTorch scripts for training a DCGAN and a small CNN classifier. `augment_dataset.py` can be used to expand a small dataset of images into a larger synthetic set.

## Usage

1. Place your images in `data/user_photos/drunk` and `data/user_photos/sober`.
2. Run `python augment_dataset.py` to train a GAN for each class and generate new images in `data/augmented/`.
3. Use the generated images to train the classifier in `classifier/train_classifier.py`.

