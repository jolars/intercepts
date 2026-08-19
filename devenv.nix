{
  inputs,
  pkgs,
  ...
}:

{
  packages = with pkgs; [
    git
    bashInteractive
    quartoMinimal
    go-task
    librsvg
    liblinear
  ];

  # These source trees let the controlled comparison run without a separate
  # mutable checkout; devenv.lock records their full revisions and NAR hashes.
  env.SKGLM_PRE_FIX_SOURCE = inputs.skglm-pre-fix;
  env.SKGLM_POST_FIX_SOURCE = inputs.skglm-post-fix;

  languages = {
    julia = {
      enable = true;
      package = (
        pkgs.julia_111-bin.withPackages [
          "Arpack"
          "CSV"
          "CairoMakie"
          "DataFrames"
          "Distributions"
          "DrWatson"
          "GLM"
          "JLD2"
          "LIBSVMdata"
          "LaTeXStrings"
          "AlgebraOfGraphics"
          "LanguageServer"
          "LinearAlgebra"
          "PkgTemplates"
          "ProjectRoot"
          "QuartoNotebookRunner"
          "Random"
          "Revise"
          "SparseArrays"
          "Test"
          "Runic"
          "Lasso"
        ]
      );
    };

    r = {
      enable = true;

      package = (
        pkgs.rWrapper.override {
          packages = with pkgs.rPackages; [
            glmnet
            ncvreg
            biglasso
            LiblineaR
            adelie
            textir
            Matrix
          ];
        }
      );
    };

    python =
      let
        python3 = pkgs.python3.override {
          packageOverrides = self: super: {
            numba = super.numba.override { cudaSupport = false; };
          };
        };

        download = (
          python3.pkgs.buildPythonPackage rec {
            pname = "download";
            version = "0.3.5";
            pyproject = true;

            src = pkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-iEqIVHWzzb7Aqid+QWQ5lcM5Sh4GSggW9T//rko4ITA=";
            };

            build-system = with python3.pkgs; [
              setuptools
              wheel
            ];

            dependencies = with python3.pkgs; [
              requests
              six
              tqdm
            ];

            pythonImportsCheck = [
              "download"
            ];
          }
        );

        libsvmdata = (
          python3.pkgs.buildPythonPackage {
            pname = "libsvmdata";
            version = "0.5.0";
            pyproject = true;

            src = pkgs.fetchFromGitHub {
              owner = "mathurinm";
              repo = "libsvmdata";
              rev = "b7cd318f3ea0e02d88791413922fc129a49953f6";
              hash = "sha256-yQHo7OwC10T5mEx8zlnwp4efB8ux+RhDqrOgGiMq4iQ=";
            };

            build-system = with python3.pkgs; [
              setuptools
            ];

            dependencies = with python3.pkgs; [
              download
              numpy
              scipy
              scikit-learn
            ];

            pythonImportsCheck = [
              "libsvmdata"
            ];
          }
        );

        celer = (
          python3.pkgs.buildPythonPackage rec {
            pname = "celer";
            version = "0.7.4";
            pyproject = true;

            src = pkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-Lj5ltSGOskVRVfPFTSxrM7SG/RE7Z/vh8xcj/h7BQ2M=";
            };

            build-system = with python3.pkgs; [
              cython
              numpy
              scipy
              setuptools
            ];

            dependencies = with python3.pkgs; [
              download
              libsvmdata
              matplotlib
              scikit-learn
              seaborn
              tqdm
              xarray
            ];

            pythonImportsCheck = [
              "celer"
            ];
          }
        );

        blitzl1 = (
          python3.pkgs.buildPythonPackage {
            pname = "blitzl1";
            version = "0.0.1";
            pyproject = true;

            src = pkgs.fetchFromGitHub {
              owner = "tbjohns";
              repo = "Blitzl1";
              rev = "aef8d02832b4c81ed3770637bf7c8ea7ee91eb45";
              hash = "sha256-NbjgXvenwtbLTz6mDt6eerdJYAcsPe2auwmc+9C3frQ=";
            };

            build-system = with python3.pkgs; [ setuptools ];

            postPatch = ''
              substituteInPlace setup.py \
                --replace-fail 'import distutils' 'import setuptools as distutils' \
                --replace-fail 'from distutils.core import setup, Extension' 'from setuptools import setup, Extension'
            '';

            dependencies = with python3.pkgs; [
              numpy
              scipy
            ];

            pythonImportsCheck = [
              "blitzl1"
            ];
          }
        );

        skglm = (
          python3.pkgs.buildPythonPackage rec {
            pname = "skglm";
            version = "0.5";
            pyproject = true;

            src = python3.pkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-EQN67BGd0oada1dLbMqLrmkJHu7v4Gy70KWqh5rXncc=";
            };

            build-system = with python3.pkgs; [ setuptools ];

            dependencies = with python3.pkgs; [
              numpy
              scikit-learn
              scipy
              numba
            ];

            doCheck = false;

            pythonImportsCheck = [
              "skglm"
            ];
          }
        );

      in
      {
        enable = true;

        package = (
          python3.withPackages (
            ps: with ps; [
              matplotlib
              numpy
              pandas
              scikit-learn
              skglm
              blitzl1
              celer
            ]
          )
        );

      };
  };

  git-hooks.hooks = {
    panache-format.enable = true;
  };
}
