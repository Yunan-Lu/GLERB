# GLERB

## Environment
python=3.7.6, numpy=1.21.6, scipy=1.7.3, pytorch=1.13.0+cpu.

## Reproducing
Change the directory to this project and run the following command in terminal.
```Terminal
python demo.py
```


## Usage
Here is a simple example of using GLERB.
```python
import numpy as np
from utils import report, binarize
from glerb import GLERB

# load data
X, D = load_dataset('sj') # this api is defined by users
L = binarize(D)

# train our model
model = GLERB().fit(X, L)

# show the recovery performance
Drec = model.label_distribution_
report(Drec, D)
```

## Datasets
- The datasets used in our work is partially provided by [PALM](http://palm.seu.edu.cn/xgeng/LDL/index.htm)
- Emotion6: [http://chenlab.ece.cornell.edu/people/kuanchuan/index.html](http://chenlab.ece.cornell.edu/people/kuanchuan/index.html)
- Twitter-LDL and Flickr-LDL: [http://47.105.62.179:8081/sentiment/index.html](http://47.105.62.179:8081/sentiment/index.html)
