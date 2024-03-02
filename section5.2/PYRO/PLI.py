class PLI:
    def __init__(self, data):
        self.data = data
        self.data_len = len(data)
        self.PLI_dict = {}
        self.root = 0
        self.node = [[]]
        self.size = [len(data)]
        self.PLI = [list(range(len(data)))]
        self.sons = [[]]
        self.cnt = 1
        for i in range(len(data[0])):
            self.attr = i
            self.pre(data, i)

    def pre(self, data, attr):
        hashcode = attr
        id = range(len(data))
        B = [x[attr] for x in data]
        result_list = [i for _, i in sorted(zip(B, id))]
        id = result_list
        ans = []
        l = 0
        while l < len(id):
            r = l
            while r + 1 < len(id) and B[id[r + 1]] == B[id[r]]:
                r += 1
            if l == r:
                l = l + 1
                continue
            kk = []
            for i in range(l, r + 1):
                kk.append(id[i])
            ans.append(kk)
            l = r + 1
        #print(attr,ans)
        self.PLI_dict[hashcode] = self.cnt
        self.node.append([attr])
        self.size.append(self.calc_size(ans))
        self.PLI.append(ans)
        self.sons[self.root].append(self.cnt)
        self.sons.append([])
        self.cnt += 1
        return self.cnt - 1

    def attr_contain(self, list1, list2):
        for i in list1:
            if not i in list2:
                return False
        return True

    def FindMin(self, id, attr_list):
        # print(id,self.node[id],attr_list)
        node_list = self.node[id]
        if not self.attr_contain(node_list, attr_list):
            return -1
        res = id
        for x in self.sons[id]:
            new_id = self.FindMin(x, attr_list)
            # print(new_id,self.size)
            if new_id == -1:
                continue
            if self.size[new_id] <= self.size[res]:
                res = new_id
        return res

    def And(self, node1, node2):
        r = [0] * self.data_len
        for i in range(len(node2)):
            for j in node2[i]:
                r[j] = i
        node3 = []
        for i in range(len(node1)):
            dict3 = {}
            for j in node1[i]:
                if r[j] == 0:
                    continue
                if not dict3.get(r[j]):
                    dict3[r[j]] = []
                dict3[r[j]].append(j)
            for j in dict3.values():
                if len(j) == 1:
                    continue
                node3.append(j)
        return node3

    def calc_size(self, ans):
        res = 0
        for i in ans:
            res += len(i)
        return res

    def calc(self, attr_list):
        hashcode = 1
        for i in attr_list:
            hashcode = (hashcode * 100 + i) % 10000000007
        if self.PLI_dict.get(hashcode):
            # print("out", attr_list)
            return self.PLI[self.PLI_dict.get(hashcode)]
        Minn_node = self.FindMin(self.root, attr_list)
        '''if attr_list==[0,14]:
            print("DA",Minn_node)'''
        new_attr_list = []
        for i in attr_list:
            if not i in self.node[Minn_node]:
                new_attr_list.append(i)
        node2 = self.calc(new_attr_list)
        ans = self.And(self.PLI[Minn_node], node2)
        self.PLI_dict[hashcode] = self.cnt
        self.node.append(attr_list)
        self.size.append(self.calc_size(ans))
        self.PLI.append(ans)
        self.sons[Minn_node].append(self.cnt)
        self.sons.append([])
        self.cnt += 1
        return self.PLI[self.cnt - 1]

    def print(self):
        q = [(0,self.root, 0)]
        head = 0
        tail = 0
        L_deep = 0
        while head <= tail:
            fa = q[head][0]
            id = q[head][1]
            deep = q[head][2]
            head += 1
            if L_deep != deep:
                print("\n")
            '''if id==0:
                break'''
            L_deep = deep
            print("DD", fa,id, self.node[id], self.size[id], end="\t")
            for i in self.sons[id]:
                q.append((id,i, deep + 1))
                tail += 1
        print(self.cnt, self.sons[self.cnt - 2])
        print("\n")


if __name__ == "__main__":
    data = [["Alex", "Smith", "m", 55302, "Brighton"], ["John", "Kramer", "m", 55301, "Brighton"],
            ["Lindsay", "Miller", "f", 55301, "Rapid Falls"], ["Alex", "Smith", "m", 55302, "Brighton"],
            ["Alex", "Miller", "f", 55301, "Brighton"]]
    P = PLI(data)
    print(P.calc([1, 2]))
