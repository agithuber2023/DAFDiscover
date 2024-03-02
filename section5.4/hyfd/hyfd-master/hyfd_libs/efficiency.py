from math import factorial as fac

def binomial(x, y):
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom

class Efficiency(object):
    def __init__(self, att, pli, window=2, comps=0, results=0.0):
        self.att = att
        self.total = sum([binomial(len(cluster), 2) for cluster in pli])
        self.window = window
        self.comps = comps
        self.results = results

        self.done = False
    def increase_comps(self):
        self.comps += 1
        if self.comps == self.total:
            self.done = True
    def eval(self):
        if self.comps == 0:
            return 0.0
        return self.results/self.comps
    def __str__(self):
        return "[a:{}|T:{}|W:{}|C:{}|R:{}|E:{}||D:{}]".format(self.att, self.total, self.window, self.comps, self.results, self.eval(), self.done)
    def __repr__(self):
        return self.__str__()
        