# Clothing Image Classification Using CNNs

**ARTE 4110U – Deep Learning Architecture & Algorithms**  
**Group 24 – G24Systems**

## Project Overview

This project uses convolutional neural networks (CNNs) to recognize three clothing attributes from colour images: category, colour, and pattern.

We're comparing three independently trained CNNs with a multi-task CNN that predicts all three attributes together. The goal is to see whether sharing learned features improves classification performance while reducing model size.

For example, the models could identify clothing as a "blue striped shirt" or "black plain T-shirt."

## Dataset

We used the [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset from Kaggle, which contains around 44,000 colour product images and their metadata.

The dataset provides category and colour labels through `articleType` and `baseColour`. Since pattern labels aren't provided directly, we identified them using keywords from product names.

For Milestone 2, we worked with 858 filtered clothing images.

| Split | Images |
|---|---:|
| Training | 600 |
| Validation | 129 |
| Testing | 129 |
| **Total** | **858** |

The experiment uses five clothing categories, five colours, and four pattern classes: plain, striped, checked, and floral.

Pattern labels haven't been fully verified manually, so the results are preliminary.

## Models and Training

We implemented two approaches using PyTorch:

- **Independent CNNs:** Three separate CNNs, each trained to predict category, colour, or pattern.
- **Multi-task CNN:** One CNN with a shared feature extractor and three output heads.

The models use three convolutional layers with 32, 64, and 128 filters, ReLU activation, max pooling, and global average pooling.

### Training Settings

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Image Size | 128 × 128 RGB |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 10 |
| Loss Function | Cross-entropy |
| Environment | Google Colab |

Training images use small rotations and translations for augmentation. Validation images aren't augmented.

## How to Run

The main experiment is in `final_milestone2 (1).ipynb`.

### 1. Open the Notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `final_milestone2 (1).ipynb` from this repository.
3. Select **Runtime → Change runtime type → T4 GPU**, if available.
4. Run the notebook cells from top to bottom.

### 2. Install Dependencies

The notebook includes a setup cell for installing the required packages.

```python
!pip -q install kagglehub pandas numpy pillow scikit-learn matplotlib torch torchvision
```

### 3. Download the Dataset

The notebook downloads the dataset using KaggleHub.

```python
import kagglehub

dataset_path = kagglehub.dataset_download(
    "paramaggarwal/fashion-product-images-small"
)
```

The images are then filtered, preprocessed, and divided into training, validation, and testing sets.

### 4. Train the Models

The notebook trains the independent CNNs and the shared multi-task CNN.

The main training call is:

```python
run('prepared', 'runs', epochs=10, batch_size=32, model='all')
```

The models are validated during training, and their best weights are saved based on validation loss.

### 5. View the Results

The notebook displays validation metrics, learning curves, confusion matrices, and sample predictions.

Experiment files are saved in the `runs/` folder. The notebook also includes an export step for downloading the results as `milestone2_results.zip`.

## Milestone 2 Results

All four models were trained for 10 epochs. The following results were recorded on the validation set.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Independent CNN – Category | 53.49% | 0.2532 |
| Independent CNN – Colour | 67.44% | 0.5788 |
| Independent CNN – Pattern | 55.81% | 0.3253 |
| Multi-task CNN – Category | 46.51% | 0.2175 |
| Multi-task CNN – Colour | 68.22% | 0.6344 |
| Multi-task CNN – Pattern | 41.09% | 0.2298 |

The multi-task CNN performed slightly better for colour prediction, while the independent models performed better for category and pattern classification.

The shared model used about 95,054 parameters, compared with approximately 281,550 parameters across the three independent CNNs combined.

These are preliminary results. Further experiments are needed to determine how consistently the models perform.

## Repository Structure

```text
fashion-mnist-robustness/
├── final_milestone2 (1).ipynb
├── README.md
├── CONTRIBUTIONS.md
├── requirements.txt
├── .gitignore
├── src/
├── tests/
└── reports/
```

| File / Folder | Description |
|---|---|
| `final_milestone2 (1).ipynb` | Main experiment, training, evaluation, and saved outputs |
| `src/` | Supplementary CNN training code |
| `tests/` | Basic tests for the supplementary implementation |
| `requirements.txt` | Required Python packages |
| `CONTRIBUTIONS.md` | Individual team contributions |
| `reports/` | Milestone 2 report and supporting files |
| `.gitignore` | Excludes datasets, temporary files, and virtual environments |

The completed notebook is the main implementation used for our reported results. The separate scripts in `src/` are supplementary and weren't used to generate those results.

## Challenges and Next Steps

One of the main challenges has been obtaining reliable pattern labels. We inferred the patterns from product descriptions, but some labels may still be incomplete or inaccurate.

The smaller dataset also makes it harder for the models to distinguish similar clothing items.

For the next stage, we plan to:

- Review and verify more pattern labels.
- Test additional training settings and augmentation methods.
- Investigate class imbalance and common classification errors.
- Run more controlled experiments using a larger dataset.
- Evaluate the final models on the held-out test set.

## Team Members

**Group 24 – G24Systems**

- Abdul Aziz Syed
- Thashon Srikantharajah
- Umair Ahmed

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for the group's individual contributions.

## References

[1] P. Aggarwal, "Fashion Product Images (Small)," Kaggle, 2019. [Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small).

[2] R. Caruana, "Multitask Learning," *Machine Learning*, vol. 28, pp. 41–75, 1997.

[3] Z. Liu et al., "DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations," *Proceedings of CVPR*, 2016.
