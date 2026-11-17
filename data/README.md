# Data

Download the dataset from Kaggle and put the CSV file(s) in this folder:

https://www.kaggle.com/datasets/pythonafroz/renewable-power-generation-and-weather-conditions

Or with the Kaggle CLI:

```bash
kaggle datasets download -d pythonafroz/renewable-power-generation-and-weather-conditions -p data --unzip
```

Raw CSV files are git-ignored. If the dataset has several files, set `CSV_FILE` in `config.py`.
