import csv
import random

with open('hughes.original.csv', 'r', encoding='utf-8-sig') as csv_file:
    csv_reader=csv.reader(csv_file)
    csv_data=[row for row in csv_reader]

#print(csv_data)
csv_file.close()
#print(csv_data)
a=[0.0000001, 0.0000001, 0.0000001, 0.005, 0.000001, 0.02, 0.01, 0.005]
n=len(csv_data)
m=len(csv_data[0])
for i in range(m):
    threshold=int(a[i]*n)
#    sum=0
    for j in range(n):
#        if sum>=threshold: break
        if j==0: continue
        if random.random()<a[i]:
            if csv_data[j][i]==csv_data[j-1][i]:
                csv_data[j][i]=csv_data[j][i]+'**'
            else:
                csv_data[j][i]=csv_data[j-1][i]
#            sum=sum+1

#print(csv_data)
with open('hughes.original-dirty.csv', mode='w', newline='', encoding='utf-8-sig') as dirtyfile:
    writer=csv.writer(dirtyfile)
    writer.writerows(csv_data)
