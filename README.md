# IteratorILP: instruction-parallel iterator reductions

[![On crates.io](https://img.shields.io/crates/v/iterator_ilp.svg)](https://crates.io/crates/iterator_ilp)
[![On docs.rs](https://docs.rs/iterator_ilp/badge.svg)](https://docs.rs/iterator_ilp/)
![Requires rustc 1.67+](https://img.shields.io/badge/rustc-1.67+-red.svg)

Ever wondered why iterator reduction methods like `sum()` are slow, or why the
generated code for searching things in your complicated iterator full of side
effects is horrible? You've come to the right place!