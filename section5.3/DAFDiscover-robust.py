import sys
import time
from fca.defs.patterns.hypergraphs import TrimmedPartitionPattern
from fca.io.transformers import List2PartitionsTransformer
from itertools import combinations
# import tane
from scipy.stats import norm
import math
from scipy.optimize import fsolve
import json

# Get reduce for Python 3+
try:
    from functools import reduce
except ImportError:
    pass

##########################################################################################
## UTILS
##########################################################################################
def read_db(path):
    hashes = {}
    with open(path, 'r', encoding='utf-8-sig') as fin:
        for t, line in enumerate(fin):
            line = line.strip()
            if line == '':
                break
            for i, s in enumerate(line.split(',')):
                hashes.setdefault(i, {}).setdefault(s, set([])).add(t)# [(i, s)] = len(hashes)
        return [PPattern.fix_desc(list(hashes[k].values())) for k in sorted(hashes.keys())], t+1

def tostr(atts):
    return ''.join([chr(65+i) for i in atts])

##########################################################################################
## CLASSES
##########################################################################################
class PPattern(TrimmedPartitionPattern):
    '''
    Represents the Stripped Partition
    '''
    @classmethod
    def intersection(cls, desc1, desc2):
        '''
        Procedure STRIPPED_PRODUCT defined in [1]
        '''
        new_desc = []
        T = {}
        S = {}
        for i, k in enumerate(desc1):
            for t in k:
                T[t] = i
            S[i] = set([])
        for i, k in enumerate(desc2):
            for t in k:
                if T.get(t, None) is not None:
                    S[T[t]].add(t)
            for t in k:
                if T.get(t, None) is not None:
                    if len(S[T[t]]) > 1:
                        new_desc.append(S[T[t]])
                    S[T[t]] = set([])
        return new_desc


class PartitionsManager(object):
    '''
    Manages the cache of already calcualted partitions.
    [1] is not very specific on how to manage partititions and memory.
    Our solution is this class where partitions are registered and cached
    TANE algorithm can access partitions using the cache.
    Partitions are calculated as an intersection or product (as mentioned in [1])
    of two partitions. This is done in GENERATE_NEXT_LEVEL when references to the
    two super-partitions are available.
    Cache is organized by levels reflecting the phases of TANE.
    Only 3 phases have to be available at all time, the current, the previous and the next.
    As such, purge_old_level purges the cache of unused levels.
    Purge cache is called at the end of each TANE phase.
    '''
    def __init__(self, T):
        '''
        Initializes the cache
        '''
        self.T = T
        self.cache = {0:None, 1:{(i,):j for i, j in enumerate(T)}}
        self.current_level = 1
    
    def new_level(self):
        '''
        Creates a cache for the new level
        '''
        self.current_level += 1
        self.cache[self.current_level] = {}
    
    def purge_old_level(self):
        '''
        Memory wipe of unused cache
        '''
        del self.cache[self.current_level-2]

    def register_partition(self, X, X0, X1):
        '''
        Registers partition of attributes in X, using partitions 
        already calculated of attributes in X0 and X1
        '''
        self.cache[len(X)][X] = PPattern.intersection(self.cache[len(X0)][X0], self.cache[len(X1)][X1])

    def check_fd(self, X, XY, r, error_list, error_tolerance):
        '''
        Main difference with [1], we do not check using procedure "e" 
        to check and FD, but we use partition subsumption
        Seems more efficient
        '''
        if not bool(X):
            return False
        # left = self.cache[len(X)][X]
        # return PPattern.leq(left, self.T[y])
        # XY=[tuple([i]) for i in set(X).union({y})]
        return bool(calculate_e(X, XY.attr_set, range(r), self) <= alpha(error_list, XY.attr_set, r, error_tolerance))

    def is_superkey(self, X):
        # return not bool(self.cache[len(X)][X])
        sum=0
        for partitions in self.cache[len(X.attr_set)][X.attr_set]:
            if len(partitions)>1: sum=sum+len(partitions)-1
        return bool(sum<=0)

class rdict(dict):
    '''
    Recursive dictionary implementing Cplus
    '''
    def __init__(self, *args, **kwargs):
        super(rdict, self).__init__(*args, **kwargs)
        self.itemlist = super(rdict, self).keys()
    def __getitem__(self, key):
        if key not in self:
            self[key] = self.recursive_search(key)
        return super(rdict, self).__getitem__(key)

    def recursive_search(self, key):
        return reduce(set.intersection, [self[tuple(key[:i]+key[i+1:])] for i in range(len(key))])

##########################################################################################
## PROCEDURES
##########################################################################################
def calculate_e(X, XA, R, checker):
    '''
    Procedure e defined in [1]
    Not in use now
    '''
    e = 0
    T = {}
    if not bool(X):
        return -1
    X = checker.cache[len(X)][X]
    XA = checker.cache[len(XA)][XA]
    
    for c in XA:
        T[next(iter(c))] = len(c)
    for c in X:
        m = 1
        for t in c:
            m = max(m, T.get(t, 0))
        e += len(c) - m
    return float(e)/len(R)

def prefix_blocks(L):
    '''
    Procedure PREFIX_BLOCKS described in [1]
    '''
    blocks = {}
    for atts in L:
        blocks.setdefault(atts.attr_set[:-1],[]).append(atts)
    return blocks.values()

def alpha(error_list, XY, r, error_tolerance):
    a=0
    for i in XY:
        c=fsolve(lambda x: norm.cdf(x)-error_tolerance[i], 0)
        a+=error_list[i]*r+c*math.sqrt(r*error_list[i]*(1-error_list[i]))
    return a/r

def probability(error_list, XY, r, error_tolerance):
    p0=1
    for i in XY:
        p0=p0*(1-error_list[i])
    p0=1-p0
    return norm.cdf((alpha(error_list, XY, r, error_tolerance)*r-r*p0)/max(math.sqrt(r*p0*(1-p0)), 1e-8))

class X_iskey:
    def __init__(self, attr) :
        self.attr_set=attr
        self.is_key=False

def new_X_valid(L, X, Cplus):
    for a, x in enumerate(X):
        check=0
        for aset in L:
            if aset.attr_set==X[:a]+X[a+1:]:
                check=1
                if aset.is_key:
                    if x in Cplus[X]: Cplus[X].remove(x)
                    map(Cplus[X].remove, filter(lambda i: i not in X, Cplus[X]))
                break
        if check==0: return False
    if not bool(Cplus[X]): return False
    return True
##########################################################################################
## TANE ALGORITHM
##########################################################################################

class DAFDiscover(object):
    '''
    As seen on TV [1]
    '''
    def __init__(self, T):
        self.T = T
        self.rules = []

        self.pmgr = PartitionsManager(T)
        self.R = range(len(T))
        

        self.Cplus = rdict()
        self.Cplus[tuple([])] = set(self.R)


    def compute_dependencies(self, L, r, error_list, error_tolerance):
        '''
        Procedure COMPUTE_DEPENDENCIES described in [1]
        '''
        for X in L:
            for y in self.Cplus[X.attr_set].intersection(X.attr_set):
                a = X.attr_set.index(y)
                LHS = X.attr_set[:a]+X.attr_set[a+1:]
                if self.pmgr.check_fd(LHS, X, r, error_list, error_tolerance):
                    self.rules.append(([list(LHS), y], list(probability(error_list, X.attr_set, r, error_tolerance)), [1-calculate_e(LHS, X.attr_set, range(r), self.pmgr)]))
                    self.Cplus[X.attr_set].remove(y)
                    map(self.Cplus[X.attr_set].remove, filter(lambda i: i not in X.attr_set, self.Cplus[X.attr_set]))

    def prune(self, L):
        '''
        Procedure PRUNE described in [1]
        '''
        clean_idx = set([])
        for X in L:
            if not bool(self.Cplus[X.attr_set]):
                clean_idx.add(X)
            if self.pmgr.is_superkey(X): # Is Superkey, since it's a stripped partition, then it's an empty set
                for y in filter(lambda x: x not in X.attr_set, self.Cplus[X.attr_set]):
                    if y in reduce(set.intersection, [self.Cplus[tuple(sorted(X.attr_set[:b]+X.attr_set[b+1:]+(y,)))] for b in range(len(X.attr_set))]):
                        #self.rules.append((X.attr_set, y))
                        self.rules.append(([list(X.attr_set), y], list(probability(error_list, X.attr_set+(y,), r, error_tolerance))))
                X.is_key=True
                # clean_idx.add(X)
        for X in clean_idx:
            L.remove(X)

    # def prefix_blocks(self, L):
    #     '''
    #     Procedure PREFIX_BLOCKS described in [1]
    #     '''
    #     blocks = {}
    #     for atts in L:
    #         blocks.setdefault(atts[:-1],[]).append(atts)
    #     return blocks.values()

    def generate_next_level(self, L):
        '''
        Procedure GENERATE_NEXT_LEVEL described in [1]
        '''
        self.pmgr.new_level()
        next_L = set([])
        for k in prefix_blocks(L):
            for i, j in combinations(k, 2):
                if i.attr_set[-1] < j.attr_set[-1]:
                    X = i.attr_set + (j.attr_set[-1],)
                else:
                    X = j.attr_set + (i.attr_set[-1],)
                # if all(X[:a]+X[a+1:] in L for a, x in enumerate(X)):
                if new_X_valid(L, X ,self.Cplus):
                    next_L.add(X_iskey(X))
                    # WE ADD THIS LINE, SEEMS A BETTER ALTERNATIVE TO CALCULATE THE PARTITION HERE WHEN
                    # WE HAVE REFERENCES TO BOTH PARTITIONS USED TO CALCULATE IT
                    self.pmgr.register_partition(X, i.attr_set, j.attr_set) 
        return next_L

    def memory_wipe(self):
        '''
        FREE SOME MEMORY!!!
        '''
        self.pmgr.purge_old_level()

    def run(self, r, error_list, error_tolerance):
        '''
        Procedure TANE in [1]
        '''
        L1 = set([X_iskey(tuple([i])) for i in self.R])
        L = [None, L1]
        l = 1 
        while bool(L[l]):
            self.compute_dependencies(L[l], r, error_list, error_tolerance)
            self.prune(L[l])
            L.append(self.generate_next_level(L[l]))
            l = l+1
            # MEMORY WIPE
            L[l-1] = None 
            self.memory_wipe()

if __name__ == "__main__":
    T , r= read_db('hughes.original-dirty.csv')
    #input_str = input("please input the error threshold of each attribute(splitted by ','):")
    #error_list = [float(item) for item in input_str.split(',')]
    error_list=[0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.01, 0.0000001, 0.0000001, 0.0000001, 0.005, 0.0000001, 0.001, 0.0000001]
    #input_str2=input("please input the error tolerance of each attribute(splitted by ','):")
    #error_tolerance=[float(item) for item in input_str2.split(',')]
    error_tolerance=[0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985]
    ntane = DAFDiscover(T)
    t0 = time.time()
    ntane.run(r, error_list, error_tolerance)
    # print(r)
    print ("\t=> Execution Time: {} seconds".format(time.time()-t0))
    print ('\t=> {} Rules Found'.format(len(ntane.rules)))
    # print(ntane.rules)
    for item in ntane.rules:
        print(item)

    #json_object=json.dump(ntane.rules, indent=4)
    with open('hughes.original-dirty-result.json', 'w') as jsonfile:
        #jsonfile.write(json_object)
        json.dump(ntane.rules, jsonfile)
    # nntane = tane.TANE(T)
    # nntane.run()
    # print ('\t=> {} Rules Found'.format(len(nntane.rules)))
    # if (ntane.rules == nntane.rules):
    #     print('right')
    # else:
    #     print('wrong')
