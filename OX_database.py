import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from datetime import date


class DissolvedOxygenDatabase(object):

    def __init__(self, **kwargs):
        self.start_date = kwargs["start_date"]
        self.database_path = kwargs["database_path"]
        self.sequence_size = kwargs["sequence_size"]
        self.train_prop = kwargs["train_prop"]

        self.start_batch_index = 0
        self.read_database()
        self.gen_train_test_set()
        self.n_train = len(self.train_data["pH"]["days"])
        self.n_test = len(self.test_data["pH"]["days"])

    def read_database(self):

        def date_to_num(date_array):
            start_date = date(self.start_date[0], self.start_date[1], self.start_date[2])
            day_array = []
            for date_string in date_array:
                current_date = date_string.split("-")
                current_date = np.array(current_date, dtype="int")
                current_date = date(current_date[0], current_date[1], current_date[2])
                days_from_start = current_date - start_date
                day_array.append(days_from_start.days)
            return day_array

        self.data_features = [f for f in listdir(self.database_path) if isfile(join(self.database_path, f))]
        self.data = {}
        for feature_file in self.data_features:
            print("loading " + feature_file)
            with open(self.database_path + feature_file, 'r') as f:
                date_array = []
                data_array = []
                for line in f:
                    if line[0] == "#":
                        continue
                    line = line.split("\t")
                    if line[0] == "agency_cd":
                        continue
                    if line[0] == "5s":
                        datetime_index = line.index("20d")
                        data_index = [index for index, elm in enumerate(line) if elm == "14n"]
                    if line[0] == "USGS":
                        date_array.append(line[datetime_index])
                        data_row = [value for i, value in enumerate(line) if i in data_index]
                        if data_row[0] == "":
                            data_array.append(np.full((1, len(data_index)), np.nan))
                            continue
                        data_array.append(np.array(data_row, dtype="float32")[np.newaxis, :])
                data_array = np.concatenate(data_array, axis=0)
                normalized_values = (data_array - np.nanmean(data_array))/np.nanstd(data_array)
                self.data[feature_file] = {"date": date_array,
                                           "values": data_array,
                                           "days": date_to_num(date_array),
                                           "normalized_values": normalized_values}

    def gen_train_test_set(self):
        self.train_data = {}
        self.test_data = {}
        for key in self.data_features:
            n_train = int(self.train_prop*len(self.data[key]["date"]))
            train_dates = self.data[key]["date"][:n_train]
            test_dates = self.data[key]["date"][n_train:]
            train_values = self.data[key]["values"][:n_train, :]
            test_values = self.data[key]["values"][n_train:, :]
            train_days = self.data[key]["days"][:n_train]
            test_days = self.data[key]["days"][n_train:]
            train_n_values = self.data[key]["normalized_values"][:n_train]
            test_n_values = self.data[key]["normalized_values"][n_train:]
            self.train_data[key] = {"date": train_dates,
                                    "values": train_values,
                                    "days": train_days,
                                    "normalized_values": train_n_values}
            self.test_data[key] = {"date": test_dates,
                                   "values": test_values,
                                   "days": test_days,
                                   "normalized_values": test_n_values}

    def plot_database(self):
        plt.figure(figsize=(13, 7))
        offset = 0
        for key in self.data.keys():
            t_serie = self.data[key]["values"][:, 0]
            days = self.data[key]["days"]
            plt.plot(days, t_serie/np.nanmax(t_serie)+offset, label=key)
            offset += 1.1
        plt.legend(loc="lower right")
        plt.xlim([0, len(self.data[key]["date"])])
        plt.title("Normalized features", fontsize=14)
        plt.xlabel("Days", fontsize=14)
        plt.savefig("database.png")

    def next_batch(self, batch_size=50, feature_list=["pH", "Temperature", "River_Discharge"]):
        """Get train batch"""
        batch = []
        target = []  # Only DO
        reset = False
        start_index = self.start_batch_index
        if batch_size == "all":
            start_index = 0
            end_index = self.n_train - 1
        else:
            end_index = self.start_batch_index + batch_size

        if end_index >= self.n_train:
            end_index = self.n_train - 1
            self.start_batch_index = 0

        for index in np.arange(start_index, end_index):
            row = []
            for feature in feature_list:
                row.append(self.train_data[feature]["normalized_values"][index, 0])
            row = np.array(row)[np.newaxis, :]
            D0_target = self.train_data["Dissolved_Oxygen"]["normalized_values"][index, 0]
            if (np.sum(np.isnan(row)) > 0) or (np.isnan(D0_target)):
                continue
            batch.append(row)
            target.append(D0_target)

        if reset:
            self.start_batch_index = 0
        else:
            self.start_batch_index += batch_size

        batch = np.concatenate(batch, axis=0)
        target = np.array(target)
        return batch, target



if __name__ == "__main__":
    path = "/home/tesla/rodrigo/machine_learning_prob/DissolvedOxygenPrediction/database/"
    sequence_size = 3
    train_prop = 0.7
    first_day = [2007, 7, 1]

    database = DissolvedOxygenDatabase(database_path=path,
                                       sequence_size=3,
                                       train_prop=train_prop,
                                       start_date=first_day)

    database.plot_database()

    print("--------- structure ----------")
    print("Database features : " + str(database.data_features))
    print("Feature attributes, database.data[\"pH\"].keys(): " + str(database.data["pH"].keys()))
    print("date: list of date strings with length "+str(len(database.data["pH"]["date"])))
    print("--------- numbers of points per feature ----------")
    for key in database.data_features:
        print("data shape in "+key+" :" +str(database.data[key]["values"].shape))
    print("--------- subsets ----------")
    print("database.train_data, database.test_data: Same format as database.data")
    print("Train Data")
    for key in database.data_features:
        print("train data shape in "+key+" :" +str(database.train_data[key]["values"].shape))
    print("Test Data")
    for key in database.data_features:
       print("test data shape in " + key + " :" + str(database.test_data[key]["values"].shape))

    for i in range(50):
        batch, target = database.next_batch()
        print("batch_shape: "+str(batch.shape))
        print("target_shape:"+str(target.shape))