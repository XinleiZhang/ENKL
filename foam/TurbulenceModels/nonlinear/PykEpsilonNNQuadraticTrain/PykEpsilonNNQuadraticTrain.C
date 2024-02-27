/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "PykEpsilonNNQuadraticTrain.H"
#include "bound.H"
#include "wallFvPatch.H"
#include "nutkWallFunctionFvPatchScalarField.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace incompressible
{
namespace RASModels
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(PykEpsilonNNQuadraticTrain, 0);
addToRunTimeSelectionTable(RASModel, PykEpsilonNNQuadraticTrain, dictionary);

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

void PykEpsilonNNQuadraticTrain::correctNut()
{
    correctNonlinearStress(fvc::grad(U_));
}


void PykEpsilonNNQuadraticTrain::correctNonlinearStress(const volTensorField& gradU)
{
    timeScale_=k_/epsilon_;

    // Quadratic (tau_NL)
    volSymmTensorField S(timeScale_*symm(gradU));
    volTensorField W(timeScale_*skew(gradU));
    theta1_ = tr(S & S);
    theta2_ = tr(W & W);

    int num_cells = this->mesh_.cells().size();
    double input_vals[num_cells][2];
    forAll(k_.internalField(), id)
    {
        input_vals[id][0] = theta1_[id];
        input_vals[id][1] = theta2_[id];
        // Info << id << ' ' << theta1_[id] << ' ' << theta2_[id] << endl;
    }

    npy_intp dim[] = {num_cells, 2};

    array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &input_vals[0]);

    PyTuple_SetItem(ml_func_args, 0, array_2d);

    pValue = (PyArrayObject *)PyObject_CallObject(ml_func, ml_func_args);

    forAll(k_.internalField(), id)
    {
        double *temp1 = (double *)PyArray_GETPTR2(pValue, id, 0);
        double *temp2 = (double *)PyArray_GETPTR2(pValue, id, 1);
        double *temp3 = (double *)PyArray_GETPTR2(pValue, id, 2);
        double *temp4 = (double *)PyArray_GETPTR2(pValue, id, 3);
        g1_[id] = (*temp1);
        g2_[id] = (*temp2);
        g3_[id] = (*temp3);
        g4_[id] = (*temp4);
        
        // cout << g2_[id] << ',' ;
        // Info << id << ' ' << g1_[id] << ' ' << g2_[id] << endl;
    }
    
    // Linear (nut)
    nut_ = -g1_*k_*timeScale_;
    nut_.correctBoundaryConditions();

    nonlinearStress_ = 
        2*k_
       *(
           g2_ * twoSymm(S&W) 
         + g3_ * dev(innerSqr(S))
         + g4_ * dev(symm(W&W))
        );
    
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

PykEpsilonNNQuadraticTrain::PykEpsilonNNQuadraticTrain
(
    const geometricOneField& alpha,
    const geometricOneField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    nonlinearEddyViscosity<incompressible::RASModel>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    Ceps1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Ceps1",
            coeffDict_,
            1.44
        )
    ),

    Ceps2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Ceps2",
            coeffDict_,
            1.92
        )
    ),

    sigmak_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "sigmak",
            coeffDict_,
            1.0
        )
    ),

    sigmaEps_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "sigmaEps",
            coeffDict_,
            1.3
        )
    ),

    k_
    (
        IOobject
        (
            IOobject::groupName("k", alphaRhoPhi.group()),
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_
    ),

    epsilon_
    (
        IOobject
        (
            IOobject::groupName("epsilon", alphaRhoPhi.group()),
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_
    ),

    g1_
    (
        IOobject
        (
            "g1",
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_
    ),

    g2_
    (
        IOobject
        (
            "g2",
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_
    ),

    g3_
    (
        IOobject
        (
            "g3",
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_
    ),

    g4_
    (
        IOobject
        (
            "g4",
            runTime_.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_
    ),

    timeScale_
    (
        IOobject
        (
            "timeScale",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedScalar("timeScale", dimTime, scalar(0.0))
    ),

    theta1_
    (
        IOobject(
            "theta1",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE),
        mesh_
    ),

    theta2_
    (
        IOobject(
            "theta2",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE),
        mesh_
    )

{
    bound(k_, kMin_);
    bound(epsilon_, epsilonMin_);

    if (type == typeName)
    {
        printCoeffs(type);
    }
    
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");
    import_array1();
    pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
    pModule = PyImport_Import(pName);
    ml_func = PyObject_GetAttrString(pModule, "ml_func");
    ml_func_args = PyTuple_New(1);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool PykEpsilonNNQuadraticTrain::read()
{
    if (nonlinearEddyViscosity<incompressible::RASModel>::read())
    {
        Ceps1_.readIfPresent(coeffDict());
        Ceps2_.readIfPresent(coeffDict());
        sigmak_.readIfPresent(coeffDict());
        sigmaEps_.readIfPresent(coeffDict());

        return true;
    }

    return false;
}


void PykEpsilonNNQuadraticTrain::correct()
{
    if (!turbulence_)
    {
        return;
    }

    nonlinearEddyViscosity<incompressible::RASModel>::correct();

    tmp<volTensorField> tgradU = fvc::grad(U_);
    const volTensorField& gradU = tgradU();

    volScalarField G
    (
        GName(),
        (nut_*twoSymm(gradU) - nonlinearStress_) && gradU
    );


    // Update epsilon and G at the wall
    epsilon_.boundaryFieldRef().updateCoeffs();

    // Dissipation equation
    tmp<fvScalarMatrix> epsEqn
    (
        fvm::ddt(epsilon_)
      + fvm::div(phi_, epsilon_)
      - fvm::laplacian(DepsilonEff(), epsilon_)
      ==
        Ceps1_*G*epsilon_/k_
      - fvm::Sp(Ceps2_*epsilon_/k_, epsilon_)
    );

    epsEqn.ref().relax();
    epsEqn.ref().boundaryManipulate(epsilon_.boundaryFieldRef());
    solve(epsEqn);
    bound(epsilon_, epsilonMin_);


    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(k_)
      + fvm::div(phi_, k_)
      - fvm::laplacian(DkEff(), k_)
      ==
        G
      - fvm::Sp(epsilon_/k_, k_)
    );

    kEqn.ref().relax();
    solve(kEqn);
    bound(k_, kMin_);


    // Re-calculate viscosity and non-linear stress
    correctNonlinearStress(gradU);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace incompressible
} // End namespace Foam

// ************************************************************************* //
