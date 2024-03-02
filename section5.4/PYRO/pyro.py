from PLI import PLI
from AFD import AFD

ccccnt = -1


def error(aFD, pil):
    if aFD.LHS == [] and ccccnt == -1:
        # print(aFD.RHS)
        L_pil = pil.calc(aFD.RHS)
        cnt = 0
        for i in L_pil:
            cnt += len(i) * len(i) - len(i)
        #return 1 - cnt / (pil.data_len * pil.data_len - pil.data_len)
        return cnt / (pil.data_len * pil.data_len - pil.data_len)

    if ccccnt == -1:
        L_pil = pil.calc(aFD.LHS)
        x = aFD.LHS+[aFD.RHS]
    else:
        L_pil = pil.calc(aFD.RHS)
        x = aFD.RHS+[ccccnt]
    R_pil = pil.calc(x)
    cnt = 0
    for i in L_pil:
        cnt += len(i) * len(i) - len(i)
    for i in R_pil:
        cnt = cnt - (len(i) * len(i) - len(i))
    '''if ccccnt==0:
        print(aFD.LHS,cnt / (pil.data_len * pil.data_len - pil.data_len))'''
    '''if cnt / (pil.data_len * pil.data_len - pil.data_len)<0.4:
        print(aFD.LHS,aFD.RHS)'''
    return cnt / (pil.data_len * pil.data_len - pil.data_len)


def my_sub(X, Y):
    Z = []
    for i in X:
        if i in Y:
            continue
        Z.append(i)
    return Z


def Hitting(S):
    from pysat.examples.hitman import Hitman
    sets = S
    res = []
    with Hitman(bootstrap_with=sets, htype='sorted') as hitman:
        for hs in hitman.enumerate():
            res.append(hs)
    return res


def find_min(X, Attr_list, P, ress):
    res = X
    for i in Attr_list:
        if i in X:
            continue
        kk = X
        kk.append(i)
        #if not (kk in rel for rel in ress):
        if True:
            if error(AFD([], kk), P) < error(AFD([], res), P):
                res = kk
    return res


def ascend(L, e_max, P, Attr_list, ress):
    #print(Attr_list)
    X = [L]
    e_X = error(AFD([], X), P)
    while True:
        if e_X <= e_max:
            break
        A = find_min(X, Attr_list, P, ress)
        if A == X:
            break
        X = A
        e_X = error(AFD([], X), P)
    return X


import queue


class Task(object):
    def __init__(self, priority, name):
        self.priority = priority
        self.name = name

    def __str__(self):
        return "Task(priority={p}, name={n})".format(p=self.priority, n=self.name)

    def __lt__(self, other):
        """ 定义<比较操作符。"""
        # 从大到小排序
        return self.priority < other.priority


def calc_subset(P, M):
    res = []
    for i in M:
        if i == P:
            continue
        if not my_sub(i, P):
            res.append(i)
    return res


def trickle_down_from(P_x, e_max, PLI_P, Attr_list):
    #print("dd",P_x,Attr_list)
    if len(P_x) > 1:
        P = queue.PriorityQueue()
        for i in P_x:
            ddx = []
            for j in P_x:
                if j == i:
                    continue
                ddx.append(j)
            P.put_nowait(Task(error(AFD([], ddx), PLI_P), ddx))
        while not P.empty():
            G_x = P.get()
            G_x, g_x = G_x.name, G_x.priority
            if g_x > e_max:
                #continue
                break
            C = trickle_down_from(G_x, e_max, PLI_P, Attr_list)
            if C != -1:
                return C
    e_x = error(AFD([], P_x), PLI_P)
    if e_x <= e_max:
        return P_x
    return -1


def Est(L, e_max, PLI_P, Attr_list):
    # print(L)
    M = []
    P = queue.PriorityQueue()
    P.put_nowait(Task(error(AFD([], L), PLI_P), L))
    while not P.empty():
        P_x = P.get()
        P_x = P_x.name
        M_x = calc_subset(P_x, M)
        if M_x:
            M_hit = Hitting(M)
            for H in M_hit:
                if error(AFD([], my_sub(P_x, H)), PLI_P)<=e_max:
                    P.put_nowait(Task(error(AFD([], my_sub(P_x, H)), PLI_P), my_sub(P_x, H)))
        else:
            m = trickle_down_from(P_x, e_max, PLI_P, Attr_list)
            if m != -1:
                M.append(m)
                #P.put_nowait(Task(error(AFD([], P_x), PLI_P), P_x))
    #print(M)
    return M


def Pyro(R, e_phi, e_v):
    pli_cache = PLI(R)
    res = []
    global ccccnt
    for i in range(-1, len(R[0])):
        error = e_v if i == -1 else e_phi
        Attr_List = []
        ccccnt = i
        ress=[]
        for j in range(len(R[0])):
            if i == j:
                continue
            Attr_List.append(j)
        for j in range(len(R[0])):
            if j == i:
                continue
            #print(Attr_List, ccccnt)
            x = ascend(j, error, pli_cache, Attr_List, ress)
            kk = Est(x, error, pli_cache, Attr_List)
            for k in kk:
                #print(k,i)
                #res.append(AFD(k, i))
                res.append((tuple(k), i))
                ress.append(k)
    return res, pli_cache


from pre import pre_glass, pre_process
from Score import get_Score

if __name__ == "__main__":
    #Data, name_list = pre_glass()
    Data1, Data2=pre_process()
    import time
    import resource

    start = time.time()
    ans1, pli1 = Pyro(Data1, 0.05, 0.05)
    end = time.time()
    #print("No.1")
    #print("Glass:", Data1.shape)
    print("\t=> Execution Time: {} seconds".format((end - start)))
    #print("Score:", get_Score(pli, ans))
    #pli.print()
    print ("\t=> Execution Memory: {} kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print("Cnt:", len(set(ans1)))
    # ans2, pli2=Pyro(Data2, 0, 0)

    # precision=1-(len(set(ans1)-set(ans2))/len(set(ans1)))
    # print("P:", precision)
    # print('hahahaha')
    # recall=1-(len(set(ans2)-set(ans1))/len(set(ans2)))
    # print("R:", recall)
    # print("F:", 2*precision*recall/(precision+recall))
    # for i in set(ans):
    #     print(i)
        #i.print()
    #print(len(pli.calc([0,14])[0])+len(pli.calc([0,14])[1]))
    '''data = [["Alex", "Smith", "m", 55302, "Brighton"], ["John", "Kramer", "m", 55301, "Brighton"],
            ["Lindsay", "Miller", "f", 55301, "Rapid Falls"], ["Alex", "Smith", "m", 55302, "Brighton"],
            ["Alex", "Miller", "f", 55301, "Brighton"]]
    P = Pyro(data, 0.1, 0.1)
    print(len(P))'''
