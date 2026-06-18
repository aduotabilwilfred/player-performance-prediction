import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../../configs"))

from read_config import get_config

path = get_config("data_path")

data = pd.read_csv(path)
print(data.head(5))

print("=" * 50)

print(len(data.columns.tolist()))
print(f"Columns: {data.columns.tolist()}")

print(data.info)
