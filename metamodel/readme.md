# Efficient learning of symbolic turbulence models via neural networks

To use this metamodel, you may need 
- [DAFI](https://github.com/XinleiZhang/ENKL/tree/master/dafi) 
- [AI Feynman](https://github.com/SJ001/AI-Feynman)
- [PyFoam](https://github.com/argonne-lcf/PythonFOAM)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenFOAM](https://www.openfoam.com/)

DAFI is used to train neural network turbulence model.
The PyFoam is used to embed neural network turbulence model by tensorflow into OpenFoam.
PFI value evaluation is then estimated by [scikit learn](https://scikit-learn.org/stable/modules/permutation_importance.html).
Finally, the AI Feynman is used to reveal symbolic expression from neural network. 

We provide two cases, the square duct flow and the periodic hill. In each folder, a DAFI trained neural network is provided, so as the input-output data table. A Python code that calculates PFI value and run AI Feynman that reveals symbolic regression is provided. 


