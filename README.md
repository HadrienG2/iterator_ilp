# IteratorILP: instruction-parallel iterator reductions

[![On crates.io](https://img.shields.io/crates/v/iterator_ilp.svg)](https://crates.io/crates/iterator_ilp)
[![On docs.rs](https://docs.rs/iterator_ilp/badge.svg)](https://docs.rs/iterator_ilp/)
[![Continuous Integration](https://github.com/HadrienG2/iterator_ilp/workflows/Continuous%20Integration/badge.svg)](https://github.com/HadrienG2/iterator_ilp/actions?query=workflow%3A%22Continuous+Integration%22)
![Requires rustc 1.66+](https://img.shields.io/badge/rustc-1.66+-red.svg)

Ever wondered why iterator reduction methods like `sum()` perform badly on
floating-point data, or why nontrivial search methods like `any()` do not
generate efficient code on iterators with side-effects? You've come to the right
place!

You can read the full story and how this crate lets you solve the problem in
[the docs.rs documentation](https://docs.rs/iterator_ilp/latest/iterator_ilp/).
