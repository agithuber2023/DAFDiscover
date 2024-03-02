# import sys
# sys.path.append('c:/users/lu_mi/appdata/local/programs/python/python37/lib/site-packages')
import numpy as np
#import Levenshtein
import csv

# def getDistanceRelation(r):
#     res = []
#     for i in range(len(r) - 1):
#         j = i + 1
#         kk = []
#         for k in range(len(r[i])):
#             kk.append(Levenshtein.distance(r[i][k], r[j][k]))
#         res.append(kk)
#     res = np.array(res)
#     return res


def pre_glass():
    fileHandler = open("data/adult.data", "r")
    data = []
    while True:
        line = fileHandler.readline()
        if not line:
            break
        x = line.split(',')
        data.append(x)

    data = np.array(data)
    Distance = data

    label = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
             "race", "sex",
             "capital-gain", "capital-gain", "hours-per-week", "native-country", "income"]
    return Distance, label

def pre_process():
    with open('PYRO/forestfires-enhanced-20-dirty.csv', 'r', encoding='utf-8-sig') as csv_file:
        csv_reader=csv.reader(csv_file)
        csv_data=[row for row in csv_reader]

    # for item in csv_data:
    #     print(item)
    csv_file.close()

    data1=np.array(csv_data)

    # with open('PYRO/forestfires-enhanced-20.csv', 'r', encoding='utf-8-sig') as csv_file:
    #     csv_reader=csv.reader(csv_file)
    #     csv_data=[row for row in csv_reader]

    # # for item in csv_data:
    # #     print(item)
    # csv_file.close()

    # data2=np.array(csv_data)

    return data1