import sklearn
from sklearn.utils import shuffle, column_or_1d
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing,  neighbors, svm
import pickle

import numpy as np
from sklearn.utils.validation import check_is_fitted

#Funkcja Pozwalająca wykarzystać nieznane wcześniej zmienne.
class TolerantLabelEncoder(preprocessing.LabelEncoder):
    def __init__(self, ignore_unknown=False,
                       unknown_original_value='unknown',
                       unknown_encoded_value=-1):
        self.ignore_unknown = ignore_unknown
        self.unknown_original_value = unknown_original_value
        self.unknown_encoded_value = unknown_encoded_value

    def transform(self, y):
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        indices = np.isin(y, self.classes_)
        if not self.ignore_unknown and not np.all(indices):
            raise ValueError("y contains new labels: %s"
                                         % str(np.setdiff1d(y, self.classes_)))

        y_transformed = np.searchsorted(self.classes_, y)
        y_transformed[~indices]=self.unknown_encoded_value
        return y_transformed

    def inverse_transform(self, y):
        check_is_fitted(self, 'classes_')

        labels = np.arange(len(self.classes_))
        indices = np.isin(y, labels)
        if not self.ignore_unknown and not np.all(indices):
            raise ValueError("y contains new labels: %s"
                                         % str(np.setdiff1d(y, self.classes_)))

        y_transformed = np.asarray(self.classes_[y], dtype=object)
        y_transformed[~indices]=self.unknown_original_value
        return y_transformed

#wczytywanie danych do tworzenia modelu
data = pd.read_csv("AttackDetection.csv", low_memory=False)

le = preprocessing.LabelEncoder()
source = le.fit_transform(list(data["Source"]))
np.save('classes1.npy', le.classes_)
Destination = le.fit_transform(list(data["Destination"]))
np.save('classes2.npy', le.classes_)
Protocol = le.fit_transform(list(data["Protocol"]))
np.save('classes3.npy', le.classes_)
Length = le.fit_transform(list(data["Length"]))
np.save('classes4.npy', le.classes_)
Type = le.fit_transform(list(data["Type"]))
np.save('classes5.npy', le.classes_)
Source_Port = le.fit_transform(list(data["Source_Port"]))
np.save('classes6.npy', le.classes_)
Desc_Port = le.fit_transform(list(data["Desc_Port"]))
np.save('classes7.npy', le.classes_)
Time_shift = le.fit_transform(list(data["Time_shift"]))
np.save('classes8.npy', le.classes_)
Stream_index = le.fit_transform(list(data["Stream_index"]))
np.save('classes9.npy', le.classes_)
InterfaceId = le.fit_transform(list(data["InterfaceId"]))
np.save('classes10.npy', le.classes_)
Time_from_prev = le.fit_transform(list(data["Time_from_previous"]))
np.save('classes11.npy', le.classes_)
Flags = le.fit_transform(list(data["Flags"]))
np.save('classes12.npy', le.classes_)
FlagsTCP = le.fit_transform(list(data["FlagsTCP"]))
np.save('classes13.npy', le.classes_)
Attack = list(data["Attack"])


predict = "Attack"
x = list(zip(source, Destination, Protocol, Length, Type, Source_Port, Desc_Port, Time_shift, Stream_index, InterfaceId, Time_from_prev, Flags, FlagsTCP))
y = list(Attack)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size =0.2 )


#poniższy zakomentowany kod tworzy i zapisuje model
#model = svm.SVC(kernel='rbf', gamma=0.9, C=1)
#model = KNeighborsClassifier(n_neighbors=7)
#model.n_jobs = -1
#model.fit(x_train, y_train)
#acc = model.score(x_test, y_test)
#print(acc)

#with open("DDOSmodel.pickle", "wb") as f:
  #  pickle.dump(model, f)

pickle_in = open("DDOSmodel.pickle", "rb")
model = pickle.load(pickle_in)

#wczytywanie danych testowych
data2 = pd.read_csv("normal2.csv", low_memory=False)

tle = TolerantLabelEncoder(ignore_unknown=True)
tle.classes_ = np.load('classes1.npy')
source2 = tle.transform(list(data2["Source"]))
tle.classes_ = np.load('classes2.npy')
Destination2 = tle.transform(list(data2["Destination"]))
tle.classes_ = np.load('classes3.npy')
Protocol2 = tle.transform(list(data2["Protocol"]))
tle.classes_ = np.load('classes4.npy')
Length2 = tle.transform(list(data2["Length"]))
tle.classes_ = np.load('classes5.npy')
Type2 = tle.transform(list(data2["Type"]))
tle.classes_ = np.load('classes6.npy')
Source_Port2 = tle.transform(list(data2["Source_Port"]))
tle.classes_ = np.load('classes7.npy')
Desc_Port2 = tle.transform(list(data2["Desc_Port"]))
tle.classes_ = np.load('classes8.npy')
Time_shift2 = tle.transform(list(data2["Time_shift"]))
tle.classes_ = np.load('classes9.npy')
Stream_index2 = tle.transform(list(data2["Stream_index"]))
tle.classes_ = np.load('classes10.npy')
InterfaceId2 = tle.transform(list(data2["InterfaceId"]))
tle.classes_ = np.load('classes11.npy')
Time_from_prev2 = tle.transform(list(data2["Time_from_previous"]))
tle.classes_ = np.load('classes12.npy')
Flags2 = tle.transform(list(data2["Flags"]))
tle.classes_ = np.load('classes13.npy')
FlagsTCP2 = tle.transform(list(data2["FlagsTCP"]))
Attack2 = list(data2["Attack"])

xTestData = list(zip(source2, Destination2, Protocol2, Length2, Type2, Source_Port2, Desc_Port2, Time_shift2, Stream_index2, InterfaceId2, Time_from_prev2, Flags2, FlagsTCP2))
yTestData = list(Attack2)

#obliczanie dokładności oraz wizualizacja predykcji
acc = model.score(xTestData, yTestData)

predictions = model.predict(xTestData)
for x in range(len(predictions)):
    print((predictions[x], xTestData[x], yTestData[x]))

print(acc)





