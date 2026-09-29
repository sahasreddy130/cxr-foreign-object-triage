# Chest X-ray foreign-object triage

Research code and aggregate results for **Resolution and Tiled Inference for Foreign-Object Triage in Chest Radiographs: A Validation-Calibrated Study on object-CXR**.

The study evaluates one YOLO26-s detector on the object-CXR dataset. It compares whole-image inference at 640, 960, and 1,280 pixels with 640-pixel tiled inference at 20% overlap. Localization is measured with the dataset challenge's FROC metric; image-level classification is measured with AUC. The held-out set is the public `dev` split, since the challenge's official test split is unavailable.

## Reported aggregate results

| Inference configuration | FROC | Image AUC |
|---|---:|---:|
| Whole image, 640 px | 0.8083 | 0.9362 |
| Whole image, 960 px | 0.7935 | 0.9295 |
| Whole image, 1,280 px | 0.7458 | 0.8985 |
| Tiled, 640 px, 20% overlap | 0.6211 | 0.8336 |

The validation-selected 95% image-sensitivity threshold for whole-image 640-pixel inference achieved 93.6% sensitivity and 67.6% specificity on the held-out set, flagging 32.4 clean images per 100. The full aggregate tables are in `results/`.

## Study limitations to keep in view

The reported values come from one saved checkpoint. Training was configured for 100 epochs but stopped during epoch 43; the evaluated best checkpoint corresponded to logged epoch 40. This repository does not include the checkpoint, raw images, individual image predictions, or the split manifest. A fresh run can therefore reproduce the workflow, but may not regenerate the exact reported predictions or metrics.

## Repository contents

- `config.py`: seed, dataset paths, model and evaluation settings.
- `ocxr.py`: annotation parsing, geometry checks, YOLO conversion, and prediction I/O.
- `prepare_data.py`: stratified split and YOLO-format data preparation.
- `train.py`: model training.
- `predict.py`: whole-image and tiled inference.
- `evaluate.py`: FROC, AUC, operating-point, and category evaluation.
- `validation_thresholds.py`: choose an image-level threshold on validation data and evaluate that frozen threshold on held-out data.
- `annotate.py`: crop preparation and category-label workflow.
- `figures.py`: regenerate study figures from the project result files.
- `results/`: aggregate evaluation tables only.

The organizer's reference `froc.py` is not redistributed here. `evaluate.py --verify-froc` can cross-check results when that script is obtained separately from the [object-CXR project](https://github.com/hlk-1135/object-CXR). The main FROC implementation used by this project is in `evaluate.py`.

## Data access

The repository does not contain radiographs, crops, or annotation worksheets. Download the object-CXR data from the [Kaggle mirror](https://www.kaggle.com/datasets/raddar/foreign-objects-in-chest-xrays) after reviewing its terms and the [dataset organizers' page](https://github.com/hlk-1135/object-CXR). The organizers describe the data as available for scientific research and non-commercial use. Do not redistribute the dataset through this repository.

## Running in Google Colab

The original pipeline expects a Colab runtime with a GPU, Google Drive mounted, and the Kaggle dataset under `/content/kag/object-CXR`.

1. Clone this repository into the project folder and install dependencies:

   ```python
   !git clone https://github.com/sahasreddy130/cxr-foreign-object-triage.git /content/cxr-code
   %cd /content/cxr-code
   !pip install -r requirements.txt
   ```

2. Add Kaggle credentials to Colab Secrets as `KAGGLE_USERNAME` and `KAGGLE_KEY`, then download the dataset:

   ```python
   from google.colab import userdata
   import os
   os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
   os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
   !kaggle datasets download -d raddar/foreign-objects-in-chest-xrays -p /content/kag --unzip
   ```

3. Mount Google Drive, then run the pipeline from the repository directory:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !python prepare_data.py
   !python train.py
   !python predict.py --weights /content/drive/MyDrive/CXR-Foreign-Objects/runs/yolo26s_640/weights/best.pt --split val --sweep
   !python predict.py --weights /content/drive/MyDrive/CXR-Foreign-Objects/runs/yolo26s_640/weights/best.pt --split test --sweep
   !python validation_thresholds.py --target-sensitivity 0.95
   !python evaluate.py --split test
   !python figures.py --split test
   ```

The `dev` split is called `test` by this pipeline for evaluation; in the paper it is described as the held-out set, not the unavailable official challenge test set. Do not use held-out labels to choose the operating threshold.

The paper reports Ultralytics 8.4.121 and PyTorch 2.11.0+cu128. `requirements.txt` gives dependency lower bounds rather than a complete lockfile, and Google Colab's preinstalled CUDA/PyTorch stack can change over time.

## License and reuse

No software license has been added yet because the four-author team has not selected one. Public visibility on GitHub does not itself grant permission to reuse or redistribute this code. The dataset has separate terms, described above. The team should agree on a code license before adding one.
