import sys

o = open(sys.argv[1], 'r').read()
files = [x.split('\t')[1] for x in o.split('\n') if x != '']
for f in files:
    print("pypy3 canonical_to_minimal.py {}".format(f))
