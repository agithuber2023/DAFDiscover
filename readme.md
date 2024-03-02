#DAFDiscover: Robust Mining Algorithm for Dynamic Approximate Functional Dependencies on Dirty Data

This repository mainly implements the discovery algorithm of DAFD (DAFDiscover) and its optimization DAFDiscover+ of *DAFDiscover: Robust Mining Algorithm for Dynamic Approximate Functional Dependencies on Dirty Data*. Given a dirty dataset which satisfies the assumptions in the paper and error thresholds of each attributes of the dataset, DAFDiscover can mine data dependencies behind the dirty dataset accurately.

Besides, to make the experiments of the paper easy to re-implement, this repository also implements MIDD and SoFD, which are two comparison methods of the experiments. More details about them could be found in *DAFDiscover: Robust Mining Algorithm for Dynamic Approximate Functional Dependencies on Dirty Data*.

##About the running environment
All experiments but the experiment in section 5.4 of the paper were conducted on a Windows machine with an 11th Gen Intel(R) Core(TM) i5-11300H CPU and 16GB of memory, running Windows10 22H2. The experiment in section 5.4 wan conducted on a virtual machine with 4GB. We have implemented all methods using python3.7.

##About the datasets
All the datasets can be downloaded by clicking on the link below. Dataset: [Abalone](https://archive.ics.uci.edu/dataset/1/abalone), [Chess](https://archive.ics.uci.edu/dataset/23/chess+king+rook+vs+king), [breast-cancer-wisconsin](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original), [Forestfires](https://archive.ics.uci.edu/dataset/162/forest+fires), [Air-quality](https://archive.ics.uci.edu/dataset/360/air+quality), [Bike-sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), [Diabetic](https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset), [Hughes](lib.stat.cmu.edu/jasadata/hughes-r), [Caulkins](lib.stat.cmu.edu/jasadata/caulkins-p), [Raisin](https://archive.ics.uci.edu/dataset/850/raisin), [bitcoinheist](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset)

We emphasize that these datasets didn’t used directly, but were enhanced before experiments. Especially, when it comes to the experiment in section 5.2, for each dataset we chose 10 or 20 tuples from the original dataset and duplicate these tuple for several times before noise injection, which ensures that datasets obtained can satisfy the assumptions in the paper. For other experiments, datasets were enhanced by noise injection only or duplicated before noise injection.

##Run the code
Since all algorithms were (re-)implemented by python, we recommend you run the python files directly through VScode except HYFD(see [HYFD](https://github.com/codocedo/hyfd)) and FDX(see [FDX](https://github.com/sis-ethz/Profiler-Public)). For DAFDiscover and DAFDiscover+, a dataset and a list(format:[error threshold of attr1, error threshold of attr2, …]) that contains the error thresholds of all attributes are needed as input of algorithms , which is fixed in source codes with T and error_list. You can change them directly into the dataset you want in the source code.

##About the code
Folders with name ‘sectionXX’ contain all algorithms and datasets used in experiments of relevant sections. Besides, folders with name ‘data-enhance’ contains all programs used to enhance datasets.
