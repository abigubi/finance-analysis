{ pkgs }: {
  deps = [
    pkgs.python310Full
    pkgs.replitPackages.prybar
    pkgs.replitPackages.stderred
  ];
}
