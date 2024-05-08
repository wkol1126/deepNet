from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd

mnist = fetch_openml('mnist_784', version=1, return_X_y=False, as_frame=True, parser='auto')
print(mnist)