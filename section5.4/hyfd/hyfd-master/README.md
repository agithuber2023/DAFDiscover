# HyFD for Python
Implementation of the HyFD functional dependency miner for python.
This implementation was created for comparison purposes.
Description of the original algorithms can be found in:

Thorsten Papenbrock and Felix Naumann. 2016. A Hybrid Approach to Functional Dependency Discovery. In Proceedings of the 2016 International Conference on Management of Data (SIGMOD '16). ACM, New York, NY, USA, 821-833. DOI: https://doi.org/10.1145/2882903.2915203

## Usage
$ python hyfd.py data/silly_example.csv

### HyFD for Python (by VC)

positional arguments:  
  db_path               path to the database . 

optional arguments:  

  -h, --help            show this help message and exit   
  -s separator, --separator separator Value separator  
  -efft efficiency threshold (between 0 and 1)  
  -lf learning factor (between 0 and 1)  
  -ift invalid fds threshold (between 0 and 1)  
