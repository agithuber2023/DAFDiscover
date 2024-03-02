import random
import csv

a=1000

with open('raisin.csv', 'r', encoding='utf-8-sig') as csv_file:
    csv_reader=csv.reader(csv_file)
    csv_data=[row for row in csv_reader]

csv_data_copy=[]
for i in range(10):
    csv_data_copy.append(csv_data[i])
    
with open('raisin-enhanced-10-10000.csv', mode='w', newline='', encoding='utf-8-sig') as file:
    writer=csv.writer(file)
    for i in range(a):
        writer.writerows(csv_data_copy)