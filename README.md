EnKL-Ensemble-based Kalman Learning for data-driven turbulence modeling
============================================

This is a forked version of [the DAFI code](https://dafi.readthedocs.io) with focus on machine learning-augmented turbulence modeling.

### Additional features 
This code is built on [the DAFI code](https://dafi.readthedocs.io) and provides the following additional features.
- The ensemble variational method is added (see Folder: test_cases/sqDuct). Details are referred to [1].
- The feature importance analysis and symbolic regression methods are added to enable learning symbolic turbulence models (see Folder: metamodel). Details are referred to [2].
- The physics-constrained machine learning method is being added to learn generalizable turbulence models across different flow scenarios(under development).

## List of publications using this code:
[1] [Qingyong Luo, Xin-Lei Zhang, Guowei He. Ensemble variational method with adaptive covariance inflation for learning neural network-based turbulence models[J]. Physics of Fluids, 2024, 36(3).](https://doi.org/10.1063/5.0199175)

[2] [Chutian Wu, Xin-Lei Zhang, Duo Xu, Guowei He, A framework for learning symbolic turbulence models from indirect observation data via neural networks and feature importance analysis[J], Journal of Computational Physics, Volume 537, 2025, 114068](https://doi.org/10.1016/j.jcp.2025.114068)
