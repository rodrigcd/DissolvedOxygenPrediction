import numpy as np
from OX_database import DissolvedOxygenDatabase
from sklearn.svm import SVR
from sklearn.metrics import matthews_corrcoef, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import gridspec

path = "/home/tesla/rodrigo/machine_learning_prob/DissolvedOxygenPrediction/database/"
sequence_size = 3
train_prop = 0.75
first_day = [2007, 7, 1]

database = DissolvedOxygenDatabase(database_path=path,
                                   sequence_size=3,
                                   train_prop=train_prop,
                                   start_date=first_day)

train_input, train_target, train_days = database.next_batch(batch_size="all")

clf = SVR(C=16, gamma=0.5)
clf.fit(train_input, train_target)
predicted_target = clf.predict(train_input)
R = np.corrcoef(x=train_target, y=predicted_target)
# Denormalize Data
DO_mean, DO_std = database.data_transformation["Dissolved_Oxygen"]
predicted_target = predicted_target*DO_std + DO_mean
train_target = train_target*DO_std + DO_mean
RMSE = np.sqrt(mean_squared_error(y_pred=predicted_target, y_true=train_target))
print("Training Results")
print("Correlation Coefficient: "+str(R[0, 1]))
print("Root Mean Square Error: "+str(RMSE))

# Plot Train results
fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])
ax0 = plt.subplot(gs[0])
ax0.plot(train_days, train_target, 'r', label="Observed")
ax0.plot(train_days, predicted_target, 'b', label="Predicted")
ax0.set_ylabel("DO mg/L", fontsize=14)
ax0.set_xlabel("Days", fontsize=14)
ax0.legend(fontsize=13)
ax0.set_xlim([np.amin(train_days), np.amax(train_days)])
plt.title("SVM Train Set, DO prediction using T, pH and Q. RMSE:"+"{0:.3f}".format(RMSE)+", R:"+"{0:.3f}".format(R[0, 1]))
ax1 = plt.subplot(gs[1])
ax1.scatter(train_target, predicted_target)
ax1.text(0, 0, "R: "+str(R[0, 1]), fontsize=14)
ax1.set_xlabel("Observed DO", fontsize=14)
ax1.set_ylabel("Predicted DO", fontsize=14)
plt.tight_layout()
plt.savefig("svm_train_results.png")

# Test data
test_input, test_target, test_days = database.next_batch(set="test")
predicted_target = clf.predict(test_input)
R = np.corrcoef(x=test_target, y=predicted_target)
# Denormalize Data
DO_mean, DO_std = database.data_transformation["Dissolved_Oxygen"]
predicted_target = predicted_target*DO_std + DO_mean
test_target = test_target*DO_std + DO_mean
RMSE = np.sqrt(mean_squared_error(y_pred=predicted_target, y_true=test_target))
print("Testing Results")
print("Correlation Coefficient: "+str(R[0, 1]))
print("Root Mean Square Error: "+str(RMSE))

# Plot Test results
fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])
ax0 = plt.subplot(gs[0])
ax0.plot(test_days, test_target, 'r', label="Observed")
ax0.plot(test_days, predicted_target, 'b', label="Predicted")
ax0.set_ylabel("DO mg/L", fontsize=14)
ax0.set_xlabel("Days", fontsize=14)
ax0.set_xlim([np.amin(test_days), np.amax(test_days)])
ax0.legend(fontsize=13)
plt.title("SVM Test Set, DO prediction using T, pH and Q. RMSE:"+"{0:.3f}".format(RMSE)+", R:"+"{0:.3f}".format(R[0, 1]))
ax1 = plt.subplot(gs[1])
ax1.scatter(test_target, predicted_target)
ax1.text(0, 0, "R: "+str(R[0, 1]), fontsize=14)
ax1.set_xlabel("Observed DO", fontsize=14)
ax1.set_ylabel("Predicted DO", fontsize=14)
plt.tight_layout()
plt.savefig("svm_test_results.png")