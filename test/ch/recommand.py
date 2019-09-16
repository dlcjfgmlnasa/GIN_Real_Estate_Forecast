import os
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import dump


data = Dataset.load_builtin('ml-100k')
train_set = data.build_full_trainset()
print(train_set)

# algorithm = SVD()
# algorithm.trtrain)
