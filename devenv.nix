{
  pkgs,
  ...
}:

{
  env.GREET = "devenv";

  packages = with pkgs; [
    git
    bashInteractive
    quartoMinimal
    go-task
    librsvg
  ];

  languages = {
    julia = {
      enable = true;
      package = (
        pkgs.julia-bin.withPackages [
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
          ];
        }
      );

      lsp.enable = true;
    };

    python =
      let
        python3 = pkgs.python3.override {
          packageOverrides = self: super: {
            numba = super.numba.override { cudaSupport = false; };
          };
        };
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

            # dontUseCmakeConfigure = true;

            # nativeBuildInputs = [
            #   pkgs.cmake
            #   pkgs.ninja
            # ];

            # buildInputs = [
            #   pkgs.eigen
            # ];

            # build-system = [
            # pkgs.python3.pkgs.scikit-build-core
            # pkgs.python3.pkgs.pybind11
            # ];

            dependencies = with python3.pkgs; [
              numpy
              scipy
              furo
            ];

            # disabledTests = [
            #   "test_cdist"
            # ];

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
            ]
          )
        );

      };
  };

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
