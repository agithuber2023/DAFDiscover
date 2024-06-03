import csv
import random

with open('day-new.csv', 'r', encoding='utf-8-sig') as csv_file:
    csv_reader=csv.reader(csv_file)
    csv_data=[row for row in csv_reader]

#print(csv_data)
csv_file.close()
#print(csv_data)
# a= [0,0,0,0,0.1,0,0,0.04,0.02] #abalone
# a=[0, 0.06, 0, 0.04, 0, 0.1, 0, 0.06, 0, 0.08, 0.02] #breast-cancer-wisconsin
# a=[0,0,0.03,0,0,0.03,0.01] #chess*10
# a=[0,0,0.06,0,0,0.06,0.02] #chess*20
# a=[0,0,0.09,0,0,0.09,0.03] #chess*30
# a=[0, 0.2, 0, 0, 0, 0.05, 0.2, 0.1] #raisin*35
# a=[0, 0.06, 0, 0, 0, 0.06, 0.2, 0.1] #raisin*30
# a=[0, 0.02, 0, 0, 0, 0.02, 0.0667, 0.0333] #raisin*10
# a=[0, 0.04, 0, 0, 0, 0.04, 0.1333, 0.0667] #raisin*20
# a=[0, 0, 0.05, 0, 0.02, 0, 0, 0.2, 0.1] #day
# a=[0, 0, 0.03, 0, 0.02, 0, 0, 0.18, 0.08] #day-new*10
a=[0, 0, 0.06, 0, 0.04, 0, 0, 0.36, 0.16] #day-new*20
# a=[0, 0, 0.09, 0, 0.06, 0, 0, 0.54, 0.24] #day-new*30
n=len(csv_data)
m=len(csv_data[0])
for i in range(m):
    if i==m-1:
        threshold=int(a[i]*n)
        sum=0
        for j in range(n):
            if sum>=threshold: break
            if j==0: continue
            if random.random()<a[i]*2:
                if csv_data[j][i]==csv_data[j-1][i]:
                    csv_data[j][i]=csv_data[j][i]+'**'
                else:
                    csv_data[j][i]=csv_data[j-1][i]
                if csv_data[j][i-1]==csv_data[j-1][i-1]:
                    csv_data[j][i-1]=csv_data[j][i-1]+'**'
                else:
                    csv_data[j][i-1]=csv_data[j-1][i-1]+'*'
                sum=sum+1

    elif i==m-2:
        threshold=int((a[i]-a[i+1])*n)
        sum=0
        for j in range(n):
            if sum>=threshold: break
            if j==0: continue
            if random.random()<a[i]*2:
                if csv_data[j][i]==csv_data[j-1][i]:
                    csv_data[j][i]=csv_data[j][i]+'**'
                else:
                    csv_data[j][i]=csv_data[j-1][i]
                sum=sum+1

    else:
        threshold=int(a[i]*n)
        sum=0
        for j in range(n):
            if sum>=threshold: break
            if j==0: continue
            if random.random()<a[i]*2:
                if csv_data[j][i]==csv_data[j-1][i]:
                    csv_data[j][i]=csv_data[j][i]+'**'
                else:
                    csv_data[j][i]=csv_data[j-1][i]
                sum=sum+1

#print(csv_data)
with open('day-new-dirty-20.csv', mode='w', newline='', encoding='utf-8-sig') as dirtyfile:
    writer=csv.writer(dirtyfile)
    writer.writerows(csv_data)
