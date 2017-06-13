import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


class DissolvedOxigenDatabase(object):

    def __init__(self, **kwargs):
        self.database_path = kwargs["database_path"]
        self.sequence_size = kwargs["sequence_size"]
        self.train_prop = kwargs["train_prop"]
        self.read_database()
        self.gen_train_test_set()

    def read_database(self):
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
                self.data[feature_file] = {"date": date_array, "values": data_array}

    def gen_train_test_set(self):
        self.train_data = {}
        self.test_data = {}
        for key in self.data_features:
            n_train = int(self.train_prop*len(self.data[key]["date"]))
            train_dates = self.data[key]["date"][:n_train]
            test_dates = self.data[key]["date"][n_train:]
            train_values = self.data[key]["values"][:n_train, :]
            test_values = self.data[key]["values"][n_train:, :]
            self.train_data[key] = {"date": train_dates, "values": train_values}
            self.test_data[key] = {"date": test_dates, "values": test_values}

    def plot_database(self):
        fig = plt.figure(figsize=(13, 7))
        offset = 0
        for key in self.data.keys():
            t_serie = self.data[key]["values"][:, 0]
            plt.plot(t_serie/np.nanmax(t_serie)+offset, label=key)
            offset += 1.1
        plt.legend(loc="lower right")
        plt.xlim([0, len(self.data[key]["date"])])
        plt.title("Normalized features", fontsize=14)
        plt.xlabel("Days", fontsize=14)
        plt.savefig("database.png")



if __name__ == "__main__":
    path = "/home/rodrigo/ml_prob/DissolvedOxygenPrediction/database/"
    sequence_size = 3
    train_prop = 0.7

    database = DissolvedOxigenDatabase(database_path=path,
                                       sequence_size=3,
                                       train_prop=train_prop)

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