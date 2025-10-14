{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.systems.url = "github:nix-systems/default";
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
    inputs.systems.follows = "systems";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.bashInteractive
            pkgs.quartoMinimal
            pkgs.go-task
            pkgs.librsvg
            (pkgs.julia-bin.withPackages [
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
            ])
          ];

          shellHook = ''
            export JULIA_PROJECT="."
          '';
        };
      }
    );
}
