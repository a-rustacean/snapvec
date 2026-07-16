{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, rust-overlay, ... }:
    {
      devShells = builtins.mapAttrs (
        system: rustPkgs:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            buildInputs = [
              (rustPkgs.rust-nightly.override {
                extensions = [
                  "rust-src"
                  "rust-analyzer"
                ];
              })
            ];
          };
        }
      ) rust-overlay.packages;

      formatter = builtins.mapAttrs (_: pkgs: pkgs.nixfmt-tree) nixpkgs.legacyPackages;
    };
}
