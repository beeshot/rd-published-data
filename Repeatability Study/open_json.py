import numpy as np
import json

with open('data/Saved_Anonymized_Fruit_Data.json', 'rb') as f:
    a = json.load(f)
    print(a)

with open('data/Saved_Anonymized_Repeatability_Data.json', 'rb') as f:
    a = json.load(f)
    print(a)