# Concrete Bridge Deck Crack Detection (SDNET2018)

Deep-learning classifier that detects cracks in concrete bridge deck images, with a web demo for uploading photos.

Author: Zerihun Waje · Civil engineer turned data scientist

## Why this project
Visual inspection of bridge decks is slow, subjective and expensive. This project tests how well lightweight CNNs (transfer learning) can flag cracked deck surfaces. It focuses on the realistic problem setting: an imbalanced dataset, where missing a crack (false negative) costs more than a false alarm.

## Dataset
**SDNET2018**, Utah State University (Dorafshan, Thomas & Maguire, 2018). It contains about 56k images of 256×256 px from decks (D), walls (W) and pavements (P).
This project uses the **deck subset only** (`D/CD` = cracked, `D/UD` = uncracked).

Download it from the official Utah State University DigitalCommons page (search "SDNET2018"). Then extract it so that `data/raw/D/CD` and `data/raw/D/UD` exist. The raw data is not committed to git.

## Project structure
```
deck-crack-detection/
├── data/
│   ├── raw/              # original SDNET2018 (git-ignored)
│   └── processed/        # train/val/test split (git-ignored)
├── notebooks/            # EDA and experiments (Colab)
├── src/
│   ├── prepare_data.py   # stratified train/val/test split
│   ├── train.py          # (step 3)
│   └── evaluate.py       # (step 3)
├── app/                  # web demo (step 4)
├── models/               # saved weights (git-ignored, released separately)
├── paper/                # literature review (step 5)
└── reports/figures/      # plots for README, paper and LinkedIn
```

## Quickstart
```bash
pip install -r requirements.txt
python src/prepare_data.py --raw data/raw --out data/processed
```

## Evaluation approach
The classes are imbalanced (~15% cracked), so accuracy is **not** the headline metric.
The project reports recall (cracked), precision, F1, PR-AUC and a confusion matrix, plus Grad-CAM to check what the model looks at.

## Roadmap
- [x] 1. Repo skeleton and data split
- [ ] 2. EDA notebook
- [ ] 3. Baseline CNN + transfer learning (MobileNetV2 / EfficientNet), class weights
- [ ] 4. Web app for crack detection, deployed free
- [ ] 5. Literature review paper (preprint)
- [ ] 6. LinkedIn launch post

## Citation (dataset)
Dorafshan, S., Thomas, R. J., & Maguire, M. (2018). SDNET2018: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks. *Data in Brief*, 21, 1664–1668.

## License
MIT (code). The dataset keeps its original license.
