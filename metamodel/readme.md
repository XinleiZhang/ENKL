# Efficient learning of symbolic turbulence models via neural networks

To use this metamodel, you may need 
- [DAFI](https://github.com/XinleiZhang/ENKL/tree/master/dafi) 
- [PyFoam](https://github.com/argonne-lcf/PythonFOAM)
- [AI Feynman](https://github.com/SJ001/AI-Feynman)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenFOAM](https://www.openfoam.com/)

The [PyFoam](https://github.com/argonne-lcf/PythonFOAM) is used to embed neural network turbulence model by tensorflow into OpenFoam. 
[DAFI](https://github.com/XinleiZhang/ENKL/tree/master/dafi) is used to train neural network turbulence model using ensemble Kalman method. 
PFI value evaluation is then estimated by [scikit learn](https://scikit-learn.org/stable/modules/permutation_importance.html).
Finally, the [AI Feynman](https://github.com/SJ001/AI-Feynman) is used to reveal symbolic expression from neural network. 

We provide two cases, the square duct flow and the periodic hill. In each folder, a DAFI trained neural network is provided, so as the input-output data table. A Python code that calculates PFI value and run AI Feynman that reveals symbolic regression is provided. 

It would be better to read this lierature before using this code. 
[Chutian Wu, Xin-Lei Zhang, Duo Xu, Guowei He, A framework for learning symbolic turbulence models from indirect observation data via neural networks and feature importance analysis, Journal of Computational Physics, Volume 537, 2025, 114068,ISSN 0021-9991](https://doi.org/10.1016/j.jcp.2025.114068)

We offer some suggestion on using the code: 
- This code consists of some Jupyter notebook cells, and they are not Python programms that can be installed.
- When estimating PFI values, it would be better to use data of training data table (subspace of NN input space), rather than sampling datas directly in input sapces. 
- You may be free to use other symbolic regression code like [PySR](https://github.com/MilesCranmer/PySR). The [AI Feynman](https://github.com/SJ001/AI-Feynman) tools have some installation issues in our practice. 
- If you insist using [AI Feynman](https://github.com/SJ001/AI-Feynman), it would be better to rescale input and output data to a close scales. 
