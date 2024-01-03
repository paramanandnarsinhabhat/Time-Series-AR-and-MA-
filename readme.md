
# Time Series Analysis: AR and MA Models

This repository contains the code and dataset for training and evaluating Autoregressive (AR) and Moving Average (MA) models on time-series data.

## Overview

The `AR and MA models.ipynb` notebook includes a step-by-step process to build AR and MA models from scratch. The process includes data loading, preprocessing, visualization, stationarity testing, and model fitting.

## Structure

```
TIME-SERIES-AR-AND-MA
│
├── data
│   ├── train
│   └── valid
│
├── myenv
│
├── notebook
│   └── AR and MA models.ipynb
│
├── scripts
│   └── time_series__AR_MA.py
│
├── .gitignore
│
└── requirements.txt
```

## Data

The data directory contains `train` and `valid` subdirectories which hold the training and validation datasets respectively, used for modeling.

## Environment

The `myenv` directory is intended for the Python virtual environment where dependencies are installed.

## Notebooks

The `notebook` directory contains the Jupyter notebook `AR and MA models.ipynb` which is the main file where the analysis and modeling are conducted.

## Scripts

The `scripts` directory holds the `time_series__AR_MA.py` script for those who prefer running the analysis as a script.

## Dependencies

The project requires the following Python libraries:

- `statsmodels`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

These can be installed using the `requirements.txt` file at the root of the project directory.

## Usage

To use this project:

1. Ensure you have all the dependencies installed.
2. Navigate to the `notebook` directory and open the `AR and MA models.ipynb` notebook.
3. Follow the instructions within the notebook for running the analysis.

## Results

The notebook will guide you through the process of model fitting and will output the Root Mean Square Error (RMSE) for both AR and MA models to evaluate their performance.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request.

## License

This project is open source and available under the [MIT License](LICENSE).
