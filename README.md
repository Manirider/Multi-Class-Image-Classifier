# Multi-Class-Image-Classifier

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![License](https://img.shields.io/github/license/Manirider/Multi-Class-Image-Classifier?style=flat-square) ![Last Commit](https://img.shields.io/github/last-commit/Manirider/Multi-Class-Image-Classifier?style=flat-square) ![Issues](https://img.shields.io/github/issues/Manirider/Multi-Class-Image-Classifier?style=flat-square)

`portfolio-project`

## Project Overview

A computer vision workspace executing multi-class image classification. The pipeline demonstrates data augmentation, convolutional network selection, and evaluation metrics for image categorization tasks.

## Core Features

- Image loading pipelines using tf.data for optimized prefetching and caching.
- Custom Convolutional Neural Network (CNN) architecture with dropout layers to prevent overfitting.
- Data augmentation steps executing random flips, rotations, and zoom adjustments.
- Comprehensive evaluation charts plotting training loss, validation curves, and confusion matrices.
- Export utilities saving models for production API deployment.

## Technical Flow & Execution

Images are loaded, augmented, and normalized. The CNN extracts visual features, maps them to classification layers, and outputs class probabilities. The pipeline tracks training performance to save the best-performing model.

## Getting Started

### Requirements

- Python 3.10 or higher
- Pip package manager

### Environment Configuration

```bash
# Clone this repository
git clone https://github.com/Manirider/Multi-Class-Image-Classifier.git
cd Multi-Class-Image-Classifier

# Create a virtual environment to manage dependencies locally
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required library dependencies
pip install -r requirements.txt
```

### Execution

```bash
python main.py
```

## Directory Layout

```
Multi-Class-Image-Classifier/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
└── (source files)
```

## Contributing to the Project

I welcome issues and pull requests to make this project better. Please see the detailed guidelines in the [Contributing Guide](CONTRIBUTING.md).

## Project License

This repository is distributed under the MIT License. For complete terms, see the [LICENSE](LICENSE) file.

Developed by [S. Manikanta Suryasai](https://github.com/Manirider)
