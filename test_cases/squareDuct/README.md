# Compiling the PysimpleFoam Solver
Please go to the following URL to download and follow the accompanying instructions to compile the PysimpleFoam solver:
https://github.com/argonne-lcf/PythonFOAM/tree/main/TurbulenceModel_Examples/PysimpleFoam

The steps are as follows:
The below steps refer to directories or files in https://github.com/argonne-lcf/PythonFOAM
1. Modify the OpenFOAM path and the relevant Python path in the `prep_env.sh` file to your own paths, then run `source prep_env.sh`.
2. Enter the `TurbulenceModel_Examples/PysimpleFoam` directory and execute `wclean && wmake` to compile PysimpleFoam.

# Compiling Turbulence Models
1. Copy the turbulence model you intend to use from the `foam\TurbulenceModels\nonlinear` directory to the `src\TurbulenceModels\incompressible\turbulentTransportModels\RAS` directory in your OpenFOAM installation.
2. Add the turbulence model to the `src\TurbulenceModels\incompressible\Make\files` file in the OpenFOAM directory. For example, if the turbulence model is `PykEpsilonNNQuadraticTrain`, you would add `$(RASModels)/PykEpsilonNNQuadraticTrain/PykEpsilonNNQuadraticTrain.C`.
3. Run `./Allwclean && ./Allwmake` in the `src\TurbulenceModels` directory of your OpenFOAM installation to compile the turbulence models.