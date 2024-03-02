import csv

with open('raisin-dirty-x/raisin-dirty.csv', 'r', encoding='utf-8-sig') as readfile:
    csv_reader=csv.reader(readfile)
    csv_data=[row for row in csv_reader]
readfile.close()

data=[]
for i in range(len(csv_data)):
    data.append(csv_data[i])
    if i==499:
        with open('raisin-dirty-x/raisin-dirty-500.csv', 'w', newline='', encoding='utf-8-sig') as writefile:
            writer=csv.writer(writefile)
            writer.writerows(data)
        writefile.close()
    elif i==999:
        with open('raisin-dirty-x/raisin-dirty-1000.csv', 'w', newline='', encoding='utf-8-sig') as writefile:
            writer=csv.writer(writefile)
            writer.writerows(data)
        writefile.close()
    elif i==1499:
        with open('raisin-dirty-x/raisin-dirty-1500.csv', 'w', newline='', encoding='utf-8-sig') as writefile:
            writer=csv.writer(writefile)
            writer.writerows(data)
        writefile.close()
    elif i==1999:
        with open('raisin-dirty-x/raisin-dirty-2000.csv', 'w', newline='', encoding='utf-8-sig') as writefile:
            writer=csv.writer(writefile)
            writer.writerows(data)
        writefile.close()
    elif i==2499:
        with open('raisin-dirty-x/raisin-dirty-2500.csv', 'w', newline='', encoding='utf-8-sig') as writefile:
            writer=csv.writer(writefile)
            writer.writerows(data)
        writefile.close()
    elif i==2999:
        with open('raisin-dirty-x/raisin-dirty-3000.csv', 'w', newline='', encoding='utf-8-sig') as writefile:
            writer=csv.writer(writefile)
            writer.writerows(data)
        writefile.close()
    elif i==3499:
        with open('raisin-dirty-x/raisin-dirty-3500.csv', 'w', newline='', encoding='utf-8-sig') as writefile:
            writer=csv.writer(writefile)
            writer.writerows(data)
        writefile.close()
    elif i==3999:
        with open('raisin-dirty-x/raisin-dirty-4000.csv', 'w', newline='', encoding='utf-8-sig') as writefile:
            writer=csv.writer(writefile)
            writer.writerows(data)
        writefile.close()
    elif i==7999:
        with open('raisin-dirty-x/raisin-dirty-8000.csv', 'w', newline='', encoding='utf-8-sig') as writefile:
            writer=csv.writer(writefile)
            writer.writerows(data)
        writefile.close()
    elif i==31999:
        with open('raisin-dirty-x/raisin-dirty-32000.csv', 'w', newline='', encoding='utf-8-sig') as writefile:
            writer=csv.writer(writefile)
            writer.writerows(data)
        writefile.close()        
    elif i==127999:
        with open('raisin-dirty-x/raisin-dirty-128000.csv', 'w', newline='', encoding='utf-8-sig') as writefile:
            writer=csv.writer(writefile)
            writer.writerows(data)
        writefile.close()