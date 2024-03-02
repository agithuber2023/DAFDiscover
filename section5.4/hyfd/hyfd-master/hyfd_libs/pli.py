
def ppli(pli, el_map=lambda s:chr(97+s)):
    # print (pli, [''.join(str(i) for i in part) for part in pli]  )
    return '|'.join(['.'.join(el_map(i) for i in part) for part in pli])

class PLI(object):
    _nrecs=None
    def __init__(self, att, partition):
        self.att = att
        self.partition = partition
    def __repr__(self):
        return ppli(self.partition, el_map=str)
    def __len__(self):
        return len(self.partition)
    def __iter__(self):
        for i in self.partition:
            yield i
    def __getitem__(self, arg):
        return self.partition[arg]
    @property
    def number_of_parts(self):
        return len(self.partition)+(PLI._nrecs - sum([len(i) for i in self.partition]))