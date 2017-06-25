import matplotlib; matplotlib.rcParams['figure.figsize'] = (8, 6)
import matplotlib.pyplot as plt
import numpy as np
import GPy
from OX_database import DissolvedOxygenDatabase

path = "/home/rodrigo/ml_prob/DissolvedOxygenPrediction/database/"
sequence_size = 3
train_prop = 0.75
first_day = [2007, 7, 1]

database = DissolvedOxygenDatabase(database_path=path,
                                   sequence_size=3,
                                   train_prop=train_prop,
                                   sequence_batch_size=50,
                                   start_date=first_day)

train_input, train_target, train_days = database.next_batch(batch_size="all")
test_input, test_target, test_days = database.next_batch(set="test")

