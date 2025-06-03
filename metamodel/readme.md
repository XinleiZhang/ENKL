# Efficient learning of symbolic turbulence models via neural networks

To use this metamodel, you may need 
- [DAFI](https://github.com/XinleiZhang/ENKL/tree/master/dafi) 
- [PythonFoam](https://github.com/argonne-lcf/PythonFOAM)
- [AI Feynman](https://github.com/SJ001/AI-Feynman)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenFOAM](https://www.openfoam.com/)

The [PythonFoam](https://github.com/argonne-lcf/PythonFOAM) is used to embed a neural network-based turbulence model by TensorFlow into OpenFoam. 
[DAFI](https://github.com/XinleiZhang/ENKL/tree/master/dafi) is used to train a neural network-based turbulence model using the ensemble Kalman method. 
PFI value evaluation is then estimated by [scikit learn](https://scikit-learn.org/stable/modules/permutation_importance.html).
Finally, the [AI Feynman](https://github.com/SJ001/AI-Feynman) is used to discover symbolic expressions from the neural network. 

We provide two cases, the flow in a square duct and the flow over periodic hills. In each folder, a DAFI-trained neural network is provided, along with the input-output data table. A Python code that calculates the PFI value and runs AI Feynman-based symbolic regression is also provided. 

We offer some suggestions on using the code: 
- This code consists of some Jupyter notebook cells, and they are not Python programs that can be installed.
- When estimating PFI values, it would be better to use a training data table (subspace of NN input space), rather than sampling data directly in the input space. 
- You may be free to use other symbolic regression code like [PySR](https://github.com/MilesCranmer/PySR).
- When using [AI Feynman](https://github.com/SJ001/AI-Feynman), it would be better to rescale input and output data to a similar scale. 

It would be better to read this literature before using this code. 
[Chutian Wu, Xin-Lei Zhang, Duo Xu, Guowei He, A framework for learning symbolic turbulence models from indirect observation data via neural networks and feature importance analysis, Journal of Computational Physics, Volume 537, 2025, 114068, ISSN 0021-9991](https://doi.org/10.1016/j.jcp.2025.114068)
