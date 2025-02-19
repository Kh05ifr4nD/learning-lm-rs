{ inputs, ... }:
{
  perSystem =
    {
      config,
      system,
      ...
    }:
    let
      llvmPkgs = pkgs.llvmPackages_latest;
      overlays = [ (import inputs.rust-overlay) ];
      pkgs = import inputs.nixpkgs {
        inherit system overlays;
      };
      rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ../rust-toolchain.toml;
    in
    {
      devShells.default =
        with pkgs;
        mkShell.override { stdenv = useMoldLinker llvmPkgs.stdenv; } {
          env = {
            LIBCLANG_PATH = "${llvmPkgs.libclang.lib}/lib";
            LD_LIBRARY_PATH = "${stdenv.cc.cc.lib}/lib";
            RUST_BACKTRACE = "full";
            RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          };
          inputsFrom = with config; [
            flake-root.devShell
            treefmt.build.devShell
          ];
          name = "learning-lm-rs";
          packages =
            with pkgs;
            (
              [
                bacon
                cargo-audit
                cargo-expand
                cargo-msrv
                cargo-nextest
                cargo-sort
                rustToolchain
              ]
              ++ [
                gitui
                just
                nixd
              ]
              ++ [
                pixi
              ]
            );
          shellHook = '''';
        };
    };
}
