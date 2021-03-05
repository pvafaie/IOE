# IOE
Github repository for the paper Multi-class imbalanced semi-supervised learning from streams through online ensembles. Please visit [this link](https://ieeexplore.ieee.org/abstract/document/9346368) for accessing the original paper. 
## Packages
* Scikit-multiflow 
* Numpy
## Setup
The experiment notebook contains the experiment on how the algorithm should work with an example data. IOE.py contains the classifier, and the prequential evaluation is implemented on the experiment.py. For the best hyper-parameters, please visit the paper. In general, the recomended hyper-parameters are:
* threshold = 0.05
* forgetting_factor = 0.91
## Citation
If our work was useful for you, please feel free to cite us with the format below.
```bibtex
@INPROCEEDINGS{9346368,
  author={P. {Vafaie} and H. {Viktor} and W. {Michalowski}},
  booktitle={2020 International Conference on Data Mining Workshops (ICDMW)}, 
  title={Multi-class imbalanced semi-supervised learning from streams through online ensembles}, 
  year={2020},
  volume={},
  number={},
  pages={867-874},
  doi={10.1109/ICDMW51313.2020.00124}}
```

