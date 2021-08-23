# Cnfpack

[![github][github-badge]][github]
[![crates.io][crate-badge]][crate]

Encoder and decoder for the Cnfpack format.

Converts between the text based DIMACS CNF file format and the compressed
binary Cnfpack format.

## Format

Cnfpack is a compressed file format for CNF formulas. Converting a DIMACS CNF
formula to and from Cnfpack maintains the order of clauses as well as the order
of literals within clauses. Comments, optional whitespace or leading zeros in
the DIMACS input are not stored.

## Example Usage

```bash
# Download example instance
wget -nv https://gbd.iti.kit.edu/file/5fb0d1f02c02c6a7fb485707b637d7e4/bvsub_12973.smt2.cnf.xz
#> 2021-08-23 17:55:24 URL:https://gbd.iti.kit.edu/file/5fb0d1f02c02c6a7fb485707b637d7e4/bvsub_12973.smt2.cnf.xz [1559552/1559552] -> "bvsub_12973.smt2.cnf.xz" [1]
# Decompress `xz` file
xz -dk bvsub_12973.smt2.cnf.xz
# Convert to `cnfpack`
cnfpack bvsub_12973.smt2.cnf bvsub_12973.smt2.cnfpack
# Check file sizes
du -bh bvsub_12973.smt2.{cnf,cnf.xz,cnfpack}
#> 20M	bvsub_12973.smt2.cnf
#> 1.5M	bvsub_12973.smt2.cnf.xz
#> 2.2K	bvsub_12973.smt2.cnfpack
# Decompress and compute GBD hash to verify the formula
cnfpack -d bvsub_12973.smt2.cnfpack | tail +2 | head -c -1 | tr '\n' ' ' | md5sum
#> 5fb0d1f02c02c6a7fb485707b637d7e4 -
# ^ Matches the hash in the download URL
```

## Install

Make sure you have a working [Rust toolchain] and then run `cargo install
cnfpack` to download, install and build the latest version. Alternatively I
also provide [binaries for some platforms].


[Rust toolchain]:https://www.rust-lang.org/tools/install
[binaries for some platforms]:https://github.com/jix/cnfpack/releases

## License

This software is available under the Zero-Clause BSD license, see
[LICENSE](LICENSE) for full licensing information.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this software by you shall be licensed as defined in
[LICENSE](LICENSE).

[github]:https://github.com/jix/cnfpack
[crate]:https://crates.io/crates/cnfpack

[github-badge]: https://img.shields.io/badge/github-jix/cnfpack-blueviolet?style=flat-square
[crate-badge]: https://img.shields.io/crates/v/cnfpack?style=flat-square

