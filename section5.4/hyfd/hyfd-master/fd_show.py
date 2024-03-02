import json
import tane

#a=[[[0, 6], [10]], [[0, 6, 1], [3, 8]], [[0, 6, 1, 7], [5]], [[0, 6, 1, 9], [4]], [[0, 1], [10]], [[0, 1, 3], [8]], [[0, 1, 3, 7], [5]], [[0, 1, 4], [3, 8, 9]], [[0, 1, 4, 7], [5]], [[0, 1, 5], [3, 8]], [[0, 1, 7], [2]], [[0, 8, 1, 7], [5]], [[0, 8, 1], [3]], [[0, 2], [10]], [[0, 2, 3, 5], [8]], [[0, 2, 4], [9]], [[0, 2, 5, 7], [8]], [[0, 3], [10]], [[0, 4], [10]], [[0, 4, 7], [9]], [[0, 5], [10]], [[0, 7], [10]], [[0, 8], [10]], [[6, 1, 2, 3, 4], [10]], [[6, 1, 2, 4, 9], [10]], [[6, 1, 2, 5, 9], [10]], [[6, 1, 2, 7], [10]], [[8, 6, 1, 2], [10]], [[6, 1, 3, 4, 9], [10]], [[6, 1, 3, 5], [10]], [[8, 6, 1, 3], [10]], [[6, 1, 4, 7], [10]], [[8, 6, 1, 5], [10]], [[6, 2, 3, 4, 7], [10]], [[6, 2, 3, 5, 8, 9], [10]], [[6, 2, 4, 5, 7], [10]], [[6, 2, 5, 7, 8], [10]], [[6, 2, 5, 7, 9], [10]], [[6, 3, 4, 7, 9], [10]], [[8, 6, 3, 4], [10]], [[6, 3, 5, 7], [10]], [[6, 5, 7, 8, 9], [10]]]
if __name__=="__main__":
    with open('json/forestfires-enhanced-20-dirty-20240223115655.json', 'r') as file:
        a=json.load(file)
    a_result=[]
    sum=0
    for i in a:
        for rhs in i[1]:
            a_result.append((tuple(i[0]),rhs))
    print(len(a_result))

    # # for tane, DAFDiscover(ntane), DAFDiscover+(ntane+)
    # TT=tane.read_db('data/breast-cancer-wisconsin-enhanced-10.csv')
    # nntane = tane.TANE(TT)
    # nntane.run()
    # print ('\t=> {} Rules Found'.format(len(nntane.rules)))
    # precision=1-(len(set(a_result)-set(nntane.rules))/len(set(a_result)))
    # recall=1-(len(set(nntane.rules)-set(a_result))/len(set(nntane.rules)))

    # for hyfd
    with open('json/forestfires-enhanced-20-20240223115731.json', 'r') as file:
        a=json.load(file)
    b_result=[]
    sum=0
    for i in a:
        for rhs in i[1]:
            b_result.append((tuple(i[0]),rhs))    
    print ('\t=> {} Rules Found'.format(len(b_result)))

    precision=1-(len(set(a_result)-set(b_result))/len(set(a_result)))
    recall=1-(len(set(b_result)-set(a_result))/len(set(b_result)))
    print("P:", precision)
    print("R:", recall)
    print("F:", 2*precision*recall/(precision+recall))