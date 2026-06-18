import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../configs"))

from read_config import get_config

data_path = get_config("data_path")

data = pd.read_csv(data_path)
print(data.head(5))

print("="*50)

print(len(data.columns.tolist()))
print(f"Columns: {data.columns.tolist()}")

print(data.info)
