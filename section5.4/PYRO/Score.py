def ok(data, rfd):
    L = rfd[0]
    R = rfd[1]
    for i in L:
        if data[i[0]] > i[1]:
            return 0
    if data[R[0]] > R[1]:
        return 1
    return 2


def error(aFD, pil):
    if aFD.RHS == -1:
        return -1

    L_pil = pil.calc(aFD.LHS)
    x = aFD.LHS+[aFD.RHS]
    R_pil = pil.calc(x)
    cnt1 = 0
    cnt2 = 0
    for i in L_pil:
        cnt1 += len(i) * len(i) - len(i)
    for i in R_pil:
        cnt2 += (len(i) * len(i) - len(i))
    '''if ccccnt==0:
        print(aFD.LHS,cnt / (pil.data_len * pil.data_len - pil.data_len))'''
    '''if cnt2 > cnt1:
        print(cnt2, cnt1, x,aFD.LHS,aFD.RHS)'''
    return 1-(cnt1-cnt2)/(pil.data_len * pil.data_len - pil.data_len)


def get_Score1(pil, aFD):
    return error(aFD, pil)


import numpy as np


def get_Score2(r, rfd):
    max_dis = np.zeros(len(r[0]))
    for i in r:
        for k in range(len(i)):
            max_dis[k] = max(max_dis[k], i[k])
    resl = 0
    L = rfd[0]
    R = rfd[1]
    cnt = 0
    for i in L:
        resl += i[1] / max_dis[i[0]]
        cnt += 0
    if cnt == 0:
        resl = 0
    resr = R[1] / max_dis[R[0]]
    return (resl + resr) / 2


def get_Score(pil, AFDs):
    res = 0
    cnt = 0
    res1 = 0
    res2 = 0
    #print(len(AFDs))
    for k in AFDs:
        # print(k.LHS,k.RHS)
        kk1 = get_Score1(pil, k)
        if kk1 == -1:
            continue
        #print(kk1)
        if kk1 > 1:
            continue
        res += kk1
        cnt += 1
    if cnt == 0:
        return 0
    return res / cnt
