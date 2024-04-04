//! This crate implements an [`Iterator`] extension that provides
//! instruction-parallel reductions
//!
//! # Motivation
//!
//! Have you ever wondered why `Iterator::sum()` performs so poorly on
//! floating-point data?
//!
//! On my machine, a benchmark of summing the elements of a `Vec<f32>` that fits
//! in the CPU's L1d cache (which is a precondition for maximal computation
//! performance) sums about a billion numbers per second. This may seem
//! reasonable until you realize that modern CPUs have multi-GHz clock rates,
//! can process [multiple instructions per clock
//! cycle](https://en.wikipedia.org/wiki/Superscalar_processor), and can sum
//! [multiple floating-point numbers per
//! instruction](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data).
//! Then you come to the realization that the orders of magnitude aren't right.
//!
//! The problem lies not in the implementation of `Iterator::sum()`, but in its
//! definition. This code...
//!
//! ```
//! # let floats = [0.0; 8192];
//! let sum = floats.iter().sum::<f32>();
//! ```
//!
//! ...correctly compiles down to the same assembly as that loop...
//!
//! ```
//! # let floats = [0.0; 8192];
//! let mut sum = 0.0;
//! for &float in &floats {
//!     sum += float;
//! }
//! ```
//!
//! ...but that loop itself isn't right for modern hardware, because it does not
//! expose enough [instruction-level parallelism
//! (ILP)](https://en.wikipedia.org/wiki/Instruction-level_parallelism).
//!
//! To give some context, the Rust compiler does not allow itself to reorder
//! floating-point operations with respect to what the user wrote. This is a
//! good thing in general because floating-point arithmetic is not
//! [associative](https://en.wikipedia.org/wiki/Associative_property), which
//! means such optimizations would make program output nondeterministic (it
//! depends on what compiler optimizations were applied) and could break the
//! [numerical stability](https://en.wikipedia.org/wiki/Numerical_stability) of
//! some trickier algorithms.
//!
//! But in the case of the loop above, it also means that whatever optimizations
//! are applied, the final code must only use one accumulator, and sum the first
//! number into that accumulator, then the second number, then the third one...
//!
//! Doing so turns our whole program into a gigantic dependency chain of
//! scalar floating-point operations, with no opportunities for compilers or
//! hardware to extract parallelism. Without parallelism, hardware capabilities
//! for small-scale parallelism go to waste, and execution speed becomes limited
//! by the CPU's addition latency (how much time it takes to compute one
//! addition) rather than its addition throughput (how many unrelated additions
//! it can compute per second).
//!
//! This problem is not unique to `Iterator::sum()`, or even to floating-point
//! data. All iterator methods that perform data reduction (take an iterator of
//! elements and produce a single result) are affected by this problem to some
//! degree. All it takes is an operation whose semantics depend on the number
//! of observed iterator items or the order in which operations are performed,
//! and the compiler will generate bad code.
//!
//! And since most of these Iterator methods are documented to perform data
//! reduction in a specific way, this problem cannot be solved by improving
//! the standard library's `Iterator` implementation, at least not without
//! breaking the API of `Iterator`, which would be Very Bad (tm) and is thus
//! highly unlikely to happen.
//!
//! # What this crate provides
//!
//! [`IteratorILP`] is an Iterator extension trait that can be implemented for
//! all iterators of known length, and provides variants of the standard
//! reduction methods with a `STREAMS` const generic parameter. By tuning up
//! this parameter, you divide the iterator reduction work across more and more
//! instruction streams, exposing more and more instruction-level parallelism
//! for the compiler and hardware to take advantage of.
//!
//! This is effectively
//! [loop unrolling](https://en.wikipedia.org/wiki/Loop_unrolling), but instead
//! of making your code unreadable by manually implementing the operation
//! yourself, you let `IteratorILP` do it for you by just providing it with the
//! tuning parameter it needs.
//!
//! So to reuse the above example...
//!
//! ```
//! # let floats = [0.0; 8192];
//! use iterator_ilp::IteratorILP;
//! let sum = floats.iter().sum_ilp::<16, f32>();
//! ```
//!
//! ...would sum a slice of floating-point numbers using 16 independent
//! instruction streams, achieving a more respectable throughput of
//! 17 billion floating-point sums per second on an Intel i9-10900 CPU running
//! at 2.5 GHz. This corresponds to 6.8 additions per CPU cycle, which is
//! reasonable when considering that the hardware can do 16 additions per second
//! on correctly aligned SIMD data, but here the data is _not_ correctly
//! aligned and hence reaching about half the hardware throughput is expected.
//!
//! # How many instruction streams do I need?
//!
//! The number of independent instruction streams you need to reach peak
//! hardware performance depends on many factors:
//!
//! - What hardware you are running on
//! - What hardware features (e.g. SIMD instruction set extensions) are used by
//!   the compiled program
//! - What type of data you are manipulating
//! - How complex your input iterator and the reduction operation are
//!
//! Further, you cannot just tune the number of streams up indefinitely because
//! managing more instruction streams requires more hardware resources (CPU
//! registers, instruction cache...), and at some point you will run out of
//! these scarce resources and your runtime performance will drop. Even before
//! that, some internal compiler optimizer code complexity threshold may be
//! reached at which point the compiler will decide to stop optimizing the code
//! of individual instruction streams as much at it would optimize that of a
//! normal reduction, which will reduce performance as well.
//!
//! To give orders of magnitude, simple reductions, like the floating point sum
//! discussed above, can benefit from being spread over 16 instruction streams
//! or more on some hardware (e.g. x86 CPUs with AVX enabled), while complex
//! reductions (e.g. those using manually vectorized data) may not benefit from
//! more than 2 streams, or could even already exhibit decreased performance in
//! that configuration.
//!
//! Since the rules for determining the right number of streams are so complex,
//! and involve information unknown to this crate, we leave it up to you to
//! tune this parameter. An empirical approach is advised: take a workload
//! whose performance you care about, benchmark it with various numbers of
//! streams on a broad range of target configurations, and find out the right
//! compromise for you (which may be a hardware-dependent compromise selected
//! via `#[cfg()]` attributes and/or runtime detection if needed).

#![cfg_attr(not(any(test, feature = "std")), no_std)]

#[cfg(doc)]
use core::iter::{Product, Sum};
use core::{
    cell::RefCell,
    iter::FusedIterator,
    ops::{Add, Mul},
};
use num_traits::{One, Zero};

/// Iterator extension that provides instruction-parallel reductions
///
/// See the [crate-level documentation](crate) for more information on what
/// instruction-level parallelism is, why it's needed, how much of it you need,
/// and why standard iterator reductions may not provide enough of it.
///
/// This trait's documentation will instead focus on how and why ILP reduction
/// semantics differ from standard reduction semantics.
///
/// # General strategy
///
/// All reductions provided by this trait use the name of the equivalent
/// reduction operation provided by the standard [`Iterator`] trait, with an
/// `_ilp` suffix that stands for Instruction-Level Parallelism and a `STREAMS`
/// const parameter that lets you tune the number of independent instruction
/// streams that you want to extract.
///
/// `STREAMS` must be at least one, but at the time of writing we cannot express
/// this at the type level, so we handle requests for 0 streams through
/// panicking instead.
///
/// ILP reductions are implemented by treating an [`Iterator`] as the
/// interleaved output of `STREAMS` different iterators, that we will call
/// streams in the following to avoid confusion:
///
/// - The first item is treated as if it were the first item of the first stream
/// - The second item is treated as if it were the first item of the second stream
/// - ...and so on until STREAMS items have been processed...
/// - Then the (STREAMS+1)-th item is treated as if it were the second
///   item of the first stream, and so on.
///
/// Each of these streams is independently processed using a close cousin of the
/// algorithm that a standard iterator reduction would use, then at the end
/// the results of individual reductions are merged into a global result.
///
/// Like all forms of parallelism, instruction-level parallelism does not
/// handle early exit very well. To avoid losing its benefits, we must read out
/// input data in groups of STREAMS elements, and only after that check if any
/// of the elements that we have read requires us to terminate iteration.
///
/// In principle, a [`FusedIterator`] bound should be enough to satisfy this
/// requirement. But in practice, good code generation could not be obtained
/// without relying on the lower bound of [`Iterator::size_hint()`] to be
/// correct for safety. This is a subset of the contract of [`TrustedLen`],
/// which, unfortunately, is unstable.
///
/// Therefore, we provide our own [`TrustedLowerBound`] unsafe trait, which we
/// implement for all standard library iterators. If you need to use
/// `iterator_ilp` with another iterator whose lower size bound you trust, you
/// can do either of the following:
///
/// - Implement [`TrustedLowerBound`] for this iterator, if it's a type that you
///   control. This is the preferred path, because it allows users to leverage
///   `iterator_ilp` without unsafe assertions about types outside of their
///   control. In an ideal world, all numerical container libraries would
///   eventually provide such implementations.
/// - Use the [`AssertLowerBoundOk`] wrapper to unsafely assert, on your side,
///   that **you** trust an iterator to have a `size_hint()` implementation that
///   provides a correct lower bound.
///
/// That's it for the general strategy, now to get into the detail of particular
/// algorithms, we must divide [`Iterator`] reductions into three categories:
///
/// - [Searches](#Searching) like [`Iterator::find()`] iterate until an item
///   matching a user-provided predicate is found, then abort iteration.
/// - [Accumulations](#Accumulating) like [`Iterator::fold()`] set up an
///   accumulator and go through the entire input iterator, combining the
///   accumulator with each item and returning the final accumulator at the end.
/// - [`Iterator::sum()`] and [`Iterator::product()`] are technically
///   accumulations, but their API is so different from that of other
///   accumulations that they [are discussed separately](#sum-and-product).
///
/// # Searching
///
/// As mentioned earlier, data must be read in groups of `STREAMS` for optimal
/// instruction level parallelism. As a result, the short-circuiting feature of
/// Rust's standard search algorithms becomes somewhat meaningless and
/// deceitful, so it was dropped and the iterator is consumed instead.
///
/// Users of iterators with side effects (e.g. [`inspect()`]) must bear in mind
/// that when using the ILP version of search routines, elements may be read
/// beyond the point where the search will terminate.
///
/// # Accumulating
///
/// The signature of [`fold_ilp()`] differs a fair bit from that of
/// [`Iterator::fold()`] because instruction-parallel accumulation requires
/// setting up `STREAMS` different accumulators at the beginning and merging
/// them into one at the end. Users are invited to read [the documentation of
/// `fold_ilp()`](IteratorILP::fold_ilp()) for more details about how its usage
/// differs from that of standard [`fold()`].
///
/// Higher-level accumulation routines that do not expose the accumulator, like
/// [`reduce_ilp()`], are not as drastically affected from an API standpoint.
/// The main thing to keep in mind when using them is that since accumulation is
/// performed in a different order, results will differ for non-associative
/// reduction functions like floating-point summation, and the provided
/// reduction callable will observe a different sequence of inputs, so it should
/// not rely on ordering of inputs for correctness.
///
/// # Sum and product
///
/// The definition of the [`Sum`] and [`Product`] traits is very high-level and
/// does not allow us to inject the right code in the right place in order to
/// achieve satisfactory code generation. Therefore, our versions of the
/// [`sum()`] and [`product()`] iterator reductions have to use completely
/// different trait bounds. For sane types, this is mostly transparent,
/// reordering of operations aside.
///
/// [`fold()`]: Iterator::fold()
/// [`fold_ilp()`]: IteratorILP::fold_ilp()
/// [`inspect()`]: Iterator::inspect()
/// [`product()`]: Iterator::product()
/// [`product_ilp()`]: IteratorILP::product_ilp()
/// [`reduce_ilp()`]: IteratorILP::reduce_ilp()
/// [`sum()`]: Iterator::sum()
/// [`sum_ilp()`]: IteratorILP::sum_ilp()
/// [`TrustedLen`]: core::iter::TrustedLen
pub trait IteratorILP: Iterator + Sized + TrustedLowerBound {
    // === Searching ===

    /// Like [`Iterator::any()`], but with multiple ILP streams and consumes the
    /// iterator
    ///
    /// See also the [general IteratorILP documentation](IteratorILP) and [its
    /// section on search routines](#Searching) in particular.
    ///
    /// # Panics
    ///
    /// - If `STREAMS` is set to 0. Need at least one instruction stream to
    ///   make progress.
    #[inline]
    fn any_ilp<const STREAMS: usize>(self, mut predicate: impl FnMut(Self::Item) -> bool) -> bool {
        assert_ne!(STREAMS, 0, "Need at least one stream to make progress");
        self.find_map_ilp::<STREAMS, _>(|item| predicate(item).then_some(true))
            .unwrap_or(false)
    }

    /// Like [`Iterator::all()`], but with multiple ILP streams and consumes the iterator
    ///
    /// See also the [general IteratorILP documentation](IteratorILP) and [its
    /// section on search routines](#Searching) in particular.
    ///
    /// # Panics
    ///
    /// - If `STREAMS` is set to 0. Need at least one instruction stream to
    ///   make progress.
    #[inline]
    fn all_ilp<const STREAMS: usize>(self, mut predicate: impl FnMut(Self::Item) -> bool) -> bool {
        assert_ne!(STREAMS, 0, "Need at least one stream to make progress");
        self.find_map_ilp::<STREAMS, _>(|item| (!predicate(item)).then_some(false))
            .unwrap_or(true)
    }

    /// Like [`Iterator::find()`], but with multiple ILP streams and consumes the iterator
    ///
    /// See also the [general IteratorILP documentation](IteratorILP) and [its
    /// section on search routines](#Searching) in particular.
    ///
    /// # Panics
    ///
    /// - If `STREAMS` is set to 0. Need at least one instruction stream to
    ///   make progress.
    #[inline]
    fn find_ilp<const STREAMS: usize>(
        self,
        mut predicate: impl FnMut(&Self::Item) -> bool,
    ) -> Option<Self::Item> {
        assert_ne!(STREAMS, 0, "Need at least one stream to make progress");

        // Map the iterator in such a way that it returns Some(item) if the item
        // matches the predicate
        let mut iter = self.map(|item| predicate(&item).then_some(item));

        // Process the regular part of the stream
        let stream_len = iter.size_hint().0 / STREAMS;
        for _ in 0..stream_len {
            // Fetch one Option<Item> per stream
            let item_opts: [Option<Self::Item>; STREAMS] =
                // SAFETY: The TrustedLowerBound contract lets us assume than
                //         the lower bound returned by size_hint is correct, and
                //         the above loop will not iterate for more than this
                //         amount of iteration, so this is trusted to be safe.
                core::array::from_fn(|_| unsafe { iter.next().unwrap_unchecked() });

            // Check if the item of interest was found
            if let Some(item) = item_opts.into_iter().flatten().next() {
                return Some(item);
            }
        }

        // Process irregular elements at the end
        iter.flatten().next()
    }

    /// Like [`Iterator::find_map()`], but with multiple ILP streams and consumes the iterator
    ///
    /// See also the [general IteratorILP documentation](IteratorILP) and [its
    /// section on search routines](#Searching) in particular.
    ///
    /// # Panics
    ///
    /// - If `STREAMS` is set to 0. Need at least one instruction stream to
    ///   make progress.
    #[inline]
    fn find_map_ilp<const STREAMS: usize, Res>(
        self,
        f: impl FnMut(Self::Item) -> Option<Res>,
    ) -> Option<Res> {
        assert_ne!(STREAMS, 0, "Need at least one stream to make progress");
        self.map(f)
            .find_ilp::<STREAMS>(|res| res.is_some())
            .flatten()
    }

    /// Like [`Iterator::position()`], but with multiple ILP streams and consumes the iterator
    ///
    /// See also the [general IteratorILP documentation](IteratorILP) and [its
    /// section on search routines](#Searching) in particular.
    ///
    /// # Panics
    ///
    /// - If `STREAMS` is set to 0. Need at least one instruction stream to
    ///   make progress.
    #[inline]
    fn position_ilp<const STREAMS: usize>(
        self,
        mut predicate: impl FnMut(Self::Item) -> bool,
    ) -> Option<usize> {
        assert_ne!(STREAMS, 0, "Need at least one stream to make progress");
        self.enumerate()
            .find_map_ilp::<STREAMS, _>(|(idx, elem)| predicate(elem).then_some(idx))
    }

    /// Like [`Iterator::rposition()`], but with multiple ILP streams and consumes the iterator
    ///
    /// See also the [general IteratorILP documentation](IteratorILP) and [its
    /// section on search routines](#Searching) in particular.
    ///
    /// # Panics
    ///
    /// - If `STREAMS` is set to 0. Need at least one instruction stream to
    ///   make progress.
    #[inline]
    fn rposition_ilp<const STREAMS: usize>(
        self,
        mut predicate: impl FnMut(Self::Item) -> bool,
    ) -> Option<usize>
    where
        Self: DoubleEndedIterator + ExactSizeIterator,
    {
        assert_ne!(STREAMS, 0, "Need at least one stream to make progress");
        self.enumerate()
            .rev()
            .find_map_ilp::<STREAMS, _>(|(idx, elem)| predicate(elem).then_some(idx))
    }

    // === Accumulating ===

    /// Like [`Iterator::fold()`], but with multiple ILP streams and thus
    /// multiple accumulators
    ///
    /// `neutral` should produce the neutral element of the computation being
    /// performed. All accumulators will be initialized using this function,
    /// and eventually merged using `merge`.
    ///
    /// Implementations of `accumulate` and `merge` should not be sensitive to
    /// the traversal order of items and accumulators, respectively.
    ///
    /// See also the [general IteratorILP documentation](IteratorILP) and [its
    /// section on accumulation](#Accumulating) in particular.
    ///
    /// # Panics
    ///
    /// - If `STREAMS` is set to 0. Need at least one instruction stream to
    ///   make progress.
    #[inline]
    fn fold_ilp<const STREAMS: usize, Acc>(
        mut self,
        mut neutral: impl FnMut() -> Acc,
        mut accumulate: impl FnMut(Acc, Self::Item) -> Acc,
        mut merge: impl FnMut(Acc, Acc) -> Acc,
    ) -> Acc {
        assert_ne!(STREAMS, 0, "Need at least one stream to make progress");

        // Set up accumulators
        let mut accumulators: [Option<Acc>; STREAMS] = core::array::from_fn(|_| Some(neutral()));
        let mut accumulate_opt = |accumulator: &mut Option<Acc>, item| {
            if let Some(prev_acc) = accumulator.take() {
                *accumulator = Some(accumulate(prev_acc, item));
            }
        };

        // Accumulate the regular part of the stream
        let stream_len = self.size_hint().0 / STREAMS;
        for _ in 0..stream_len {
            for acc in &mut accumulators {
                // SAFETY: The TrustedLowerBound contract lets us assume than
                //         the lower bound returned by size_hint is correct, and
                //         the above loop will not iterate for more than this
                //         amount of iteration, so this is trusted to be safe.
                accumulate_opt(acc, unsafe { self.next().unwrap_unchecked() });
            }
        }

        // Merge the accumulators
        let mut stride = STREAMS;
        while stride > 1 {
            stride = stride / 2 + (stride % 2 != 0) as usize;
            for i in 0..stride.min(STREAMS - stride) {
                accumulators[i] = Some(merge(
                    accumulators[i].take().unwrap(),
                    accumulators[i + stride].take().unwrap(),
                ));
            }
        }
        let ilp_result = accumulators[0].take().unwrap();

        // Accumulate remaining irregular elements using standard iterator fold,
        // then merge (doing it like this improves floating-point accuracy)
        merge(ilp_result, self.fold(neutral(), accumulate))
    }

    /// Like [`Iterator::reduce()`], but with multiple ILP streams
    ///
    /// Implementations of `reduce` should not be sensitive to the order in
    /// which iterator items are traversed.
    ///
    /// See also the [general IteratorILP documentation](IteratorILP) and [its
    /// section on accumulation](#Accumulating) in particular.
    ///
    /// # Panics
    ///
    /// - If `STREAMS` is set to 0. Need at least one instruction stream to
    ///   make progress.
    #[inline]
    fn reduce_ilp<const STREAMS: usize>(
        self,
        reduce: impl FnMut(Self::Item, Self::Item) -> Self::Item,
    ) -> Option<Self::Item> {
        assert_ne!(STREAMS, 0, "Need at least one stream to make progress");
        let reduce = RefCell::new(reduce);
        self.fold_ilp::<STREAMS, _>(
            || None,
            |acc_opt, item| {
                Some(if let Some(acc) = acc_opt {
                    reduce.borrow_mut()(acc, item)
                } else {
                    item
                })
            },
            |acc_opt_1, acc_opt_2| match (acc_opt_1, acc_opt_2) {
                (Some(a), Some(b)) => Some(reduce.borrow_mut()(a, b)),
                (Some(a), _) | (_, Some(a)) => Some(a),
                (None, None) => None,
            },
        )
    }

    // === Sum and product ===

    /// Like [`Iterator::sum()`], but with multiple ILP streams, and uses
    /// different trait bounds.
    ///
    /// See also the [general IteratorILP documentation](IteratorILP) and [its
    /// section on sum and product](#sum-and-product) in particular.
    ///
    /// # Panics
    ///
    /// - If `STREAMS` is set to 0. Need at least one instruction stream to
    ///   make progress.
    #[inline(always)]
    fn sum_ilp<const STREAMS: usize, S: Add<Self::Item, Output = S> + Add<S> + Zero>(self) -> S {
        assert_ne!(STREAMS, 0, "Need at least one stream to make progress");
        self.fold_ilp::<STREAMS, _>(|| S::zero(), |acc, it| acc + it, |acc1, acc2| acc1 + acc2)
    }

    /// Like [`Iterator::product()`], but with multiple ILP streams, and uses
    /// different trait bounds.
    ///
    /// See also the [general IteratorILP documentation](IteratorILP) and [its
    /// section on sum and product](#sum-and-product) in particular.
    ///
    /// # Panics
    ///
    /// - If `STREAMS` is set to 0. Need at least one instruction stream to
    ///   make progress.
    #[inline]
    fn product_ilp<const STREAMS: usize, P: Mul<Self::Item, Output = P> + Mul<P> + One>(self) -> P {
        assert_ne!(STREAMS, 0, "Need at least one stream to make progress");
        self.fold_ilp::<STREAMS, _>(|| P::one(), |acc, it| acc * it, |acc1, acc2| acc1 * acc2)
    }
}

impl<I: Iterator + Sized + TrustedLowerBound> IteratorILP for I {}

/// An iterator that reports an accurate lower bound using [`size_hint()`]
///
/// # Safety
///
/// The lower bound reported by this iterator is guaranteed to be accurate, in
/// the sense that the iterator cannot output less items. Unsafe code can rely
/// on this being correct for safety.
///
/// For optimal performance, the lower bound should also be exact (i.e. equal to
/// the number of elements being returned) whenever possible, but this is not a
/// safety-critical property.
///
/// Since this iterator trait is a subset of the unstable [`TrustedLen`]
/// trait, it will be implemented for all implementations of [`TrustedLen`] as
/// they stabilize.
///
/// [`size_hint()`]: Iterator::size_hint()
/// [`TrustedLen`]: core::iter::TrustedLen
pub unsafe trait TrustedLowerBound: Iterator {}
//
// SAFETY: All Iterator impls from std are trusted to be implemented correctly,
//         since if you can't trust std, there is no hope for you...
mod core_iters {
    use crate::TrustedLowerBound;
    use core::{
        iter::{
            Chain, Cloned, Copied, Cycle, Empty, Enumerate, Filter, FilterMap, FlatMap, Flatten,
            FromFn, Fuse, Inspect, Map, MapWhile, Once, OnceWith, Peekable, Repeat, RepeatWith,
            Rev, Scan, Skip, SkipWhile, StepBy, Successors, Take, TakeWhile, Zip,
        },
        ops::{Range, RangeFrom, RangeInclusive},
        str::{CharIndices, Chars, EncodeUtf16, SplitAsciiWhitespace, SplitWhitespace},
    };

    unsafe impl<'a, I> TrustedLowerBound for &'a mut I where I: TrustedLowerBound + ?Sized {}
    unsafe impl<A, B> TrustedLowerBound for Chain<A, B>
    where
        A: TrustedLowerBound,
        B: TrustedLowerBound<Item = <A as Iterator>::Item>,
    {
    }
    unsafe impl TrustedLowerBound for CharIndices<'_> {}
    unsafe impl TrustedLowerBound for Chars<'_> {}
    unsafe impl<'a, I, T> TrustedLowerBound for Cloned<I>
    where
        I: TrustedLowerBound<Item = &'a T>,
        T: 'a + Clone,
    {
    }
    unsafe impl<'a, I, T> TrustedLowerBound for Copied<I>
    where
        I: TrustedLowerBound<Item = &'a T>,
        T: 'a + Copy,
    {
    }
    unsafe impl<I> TrustedLowerBound for Cycle<I> where I: TrustedLowerBound + Clone {}
    unsafe impl<T> TrustedLowerBound for Empty<T> {}
    unsafe impl TrustedLowerBound for EncodeUtf16<'_> {}
    unsafe impl<I> TrustedLowerBound for Enumerate<I> where I: TrustedLowerBound {}
    unsafe impl<I, P> TrustedLowerBound for Filter<I, P>
    where
        I: TrustedLowerBound,
        P: FnMut(&<I as Iterator>::Item) -> bool,
    {
    }
    unsafe impl<B, I, F> TrustedLowerBound for FilterMap<I, F>
    where
        F: FnMut(<I as Iterator>::Item) -> Option<B>,
        I: TrustedLowerBound,
    {
    }
    unsafe impl<I, U> TrustedLowerBound for Flatten<I>
    where
        I: TrustedLowerBound,
        <I as Iterator>::Item: IntoIterator<IntoIter = U, Item = <U as Iterator>::Item>,
        U: TrustedLowerBound,
    {
    }
    unsafe impl<I, U, F> TrustedLowerBound for FlatMap<I, U, F>
    where
        I: TrustedLowerBound,
        U: IntoIterator,
        <U as IntoIterator>::IntoIter: TrustedLowerBound,
        F: FnMut(<I as Iterator>::Item) -> U,
    {
    }
    unsafe impl<T, F> TrustedLowerBound for FromFn<F> where F: FnMut() -> Option<T> {}
    unsafe impl<I> TrustedLowerBound for Fuse<I> where I: TrustedLowerBound {}
    unsafe impl<I, F> TrustedLowerBound for Inspect<I, F>
    where
        I: TrustedLowerBound,
        F: FnMut(&<I as Iterator>::Item),
    {
    }
    unsafe impl<B, I, F> TrustedLowerBound for Map<I, F>
    where
        F: FnMut(<I as Iterator>::Item) -> B,
        I: TrustedLowerBound,
    {
    }
    unsafe impl<B, I, F> TrustedLowerBound for MapWhile<I, F>
    where
        F: FnMut(<I as Iterator>::Item) -> Option<B>,
        I: TrustedLowerBound,
    {
    }
    unsafe impl<T> TrustedLowerBound for Once<T> {}
    unsafe impl<A, F> TrustedLowerBound for OnceWith<F> where F: FnOnce() -> A {}
    unsafe impl<I> TrustedLowerBound for Peekable<I> where I: TrustedLowerBound {}
    unsafe impl TrustedLowerBound for Range<usize> {}
    unsafe impl TrustedLowerBound for Range<isize> {}
    unsafe impl TrustedLowerBound for Range<u8> {}
    unsafe impl TrustedLowerBound for Range<i8> {}
    unsafe impl TrustedLowerBound for Range<u16> {}
    unsafe impl TrustedLowerBound for Range<i16> {}
    unsafe impl TrustedLowerBound for Range<u32> {}
    unsafe impl TrustedLowerBound for Range<i32> {}
    unsafe impl TrustedLowerBound for Range<i64> {}
    unsafe impl TrustedLowerBound for Range<u64> {}
    unsafe impl TrustedLowerBound for RangeFrom<usize> {}
    unsafe impl TrustedLowerBound for RangeFrom<isize> {}
    unsafe impl TrustedLowerBound for RangeFrom<u8> {}
    unsafe impl TrustedLowerBound for RangeFrom<i8> {}
    unsafe impl TrustedLowerBound for RangeFrom<u16> {}
    unsafe impl TrustedLowerBound for RangeFrom<i16> {}
    unsafe impl TrustedLowerBound for RangeFrom<u32> {}
    unsafe impl TrustedLowerBound for RangeFrom<i32> {}
    unsafe impl TrustedLowerBound for RangeFrom<i64> {}
    unsafe impl TrustedLowerBound for RangeFrom<u64> {}
    unsafe impl TrustedLowerBound for RangeInclusive<usize> {}
    unsafe impl TrustedLowerBound for RangeInclusive<isize> {}
    unsafe impl TrustedLowerBound for RangeInclusive<u8> {}
    unsafe impl TrustedLowerBound for RangeInclusive<i8> {}
    unsafe impl TrustedLowerBound for RangeInclusive<u16> {}
    unsafe impl TrustedLowerBound for RangeInclusive<i16> {}
    unsafe impl TrustedLowerBound for RangeInclusive<u32> {}
    unsafe impl TrustedLowerBound for RangeInclusive<i32> {}
    unsafe impl TrustedLowerBound for RangeInclusive<i64> {}
    unsafe impl TrustedLowerBound for RangeInclusive<u64> {}
    unsafe impl<A: Clone> TrustedLowerBound for Repeat<A> {}
    unsafe impl<A, F> TrustedLowerBound for RepeatWith<F> where F: FnMut() -> A {}
    unsafe impl<I> TrustedLowerBound for Rev<I> where I: TrustedLowerBound + DoubleEndedIterator {}
    unsafe impl<B, I, St, F> TrustedLowerBound for Scan<I, St, F>
    where
        F: FnMut(&mut St, <I as Iterator>::Item) -> Option<B>,
        I: TrustedLowerBound,
    {
    }
    unsafe impl<I> TrustedLowerBound for Skip<I> where I: TrustedLowerBound {}
    unsafe impl<I, P> TrustedLowerBound for SkipWhile<I, P>
    where
        I: TrustedLowerBound,
        P: FnMut(&<I as Iterator>::Item) -> bool,
    {
    }
    unsafe impl TrustedLowerBound for SplitAsciiWhitespace<'_> {}
    unsafe impl TrustedLowerBound for SplitWhitespace<'_> {}
    unsafe impl<I> TrustedLowerBound for StepBy<I> where I: TrustedLowerBound {}
    unsafe impl<T, F> TrustedLowerBound for Successors<T, F> where F: FnMut(&T) -> Option<T> {}
    unsafe impl<I> TrustedLowerBound for Take<I> where I: TrustedLowerBound {}
    unsafe impl<I, P> TrustedLowerBound for TakeWhile<I, P>
    where
        I: TrustedLowerBound,
        P: FnMut(&<I as Iterator>::Item) -> bool,
    {
    }
    unsafe impl<A, B> TrustedLowerBound for Zip<A, B>
    where
        A: TrustedLowerBound,
        B: TrustedLowerBound,
    {
    }
    unsafe impl<T, const N: usize> TrustedLowerBound for core::array::IntoIter<T, N> {}
    unsafe impl TrustedLowerBound for core::ascii::EscapeDefault {}
    unsafe impl<I> TrustedLowerBound for core::char::DecodeUtf16<I> where
        I: TrustedLowerBound<Item = u16>
    {
    }
    unsafe impl TrustedLowerBound for core::char::EscapeDebug {}
    unsafe impl TrustedLowerBound for core::char::EscapeDefault {}
    unsafe impl TrustedLowerBound for core::char::EscapeUnicode {}
    unsafe impl TrustedLowerBound for core::char::ToLowercase {}
    unsafe impl TrustedLowerBound for core::char::ToUppercase {}
    unsafe impl<'a, A> TrustedLowerBound for core::option::Iter<'a, A> {}
    unsafe impl<'a, A> TrustedLowerBound for core::option::IterMut<'a, A> {}
    unsafe impl<A> TrustedLowerBound for core::option::IntoIter<A> {}
    unsafe impl<'a, A> TrustedLowerBound for core::result::Iter<'a, A> {}
    unsafe impl<'a, A> TrustedLowerBound for core::result::IterMut<'a, A> {}
    unsafe impl<A> TrustedLowerBound for core::result::IntoIter<A> {}
    unsafe impl<T> TrustedLowerBound for core::slice::Chunks<'_, T> {}
    unsafe impl<T> TrustedLowerBound for core::slice::ChunksExact<'_, T> {}
    unsafe impl<T> TrustedLowerBound for core::slice::ChunksExactMut<'_, T> {}
    unsafe impl<T> TrustedLowerBound for core::slice::ChunksMut<'_, T> {}
    unsafe impl TrustedLowerBound for core::slice::EscapeAscii<'_> {}
    unsafe impl<'a, T> TrustedLowerBound for core::slice::Iter<'a, T> {}
    unsafe impl<'a, T> TrustedLowerBound for core::slice::IterMut<'a, T> {}
    unsafe impl<T> TrustedLowerBound for core::slice::RChunks<'_, T> {}
    unsafe impl<T> TrustedLowerBound for core::slice::RChunksExact<'_, T> {}
    unsafe impl<T> TrustedLowerBound for core::slice::RChunksExactMut<'_, T> {}
    unsafe impl<T> TrustedLowerBound for core::slice::RChunksMut<'_, T> {}
    unsafe impl<'a, T, P> TrustedLowerBound for core::slice::RSplit<'a, T, P> where P: FnMut(&T) -> bool {}
    unsafe impl<'a, T, P> TrustedLowerBound for core::slice::RSplitMut<'a, T, P> where
        P: FnMut(&T) -> bool
    {
    }
    unsafe impl<'a, T, P> TrustedLowerBound for core::slice::RSplitN<'a, T, P> where P: FnMut(&T) -> bool
    {}
    unsafe impl<'a, T, P> TrustedLowerBound for core::slice::RSplitNMut<'a, T, P> where
        P: FnMut(&T) -> bool
    {
    }
    unsafe impl<'a, T, P> TrustedLowerBound for core::slice::Split<'a, T, P> where P: FnMut(&T) -> bool {}
    unsafe impl<'a, T, P> TrustedLowerBound for core::slice::SplitInclusive<'a, T, P> where
        P: FnMut(&T) -> bool
    {
    }
    unsafe impl<'a, T, P> TrustedLowerBound for core::slice::SplitInclusiveMut<'a, T, P> where
        P: FnMut(&T) -> bool
    {
    }
    unsafe impl<'a, T, P> TrustedLowerBound for core::slice::SplitMut<'a, T, P> where
        P: FnMut(&T) -> bool
    {
    }
    unsafe impl<'a, T, P> TrustedLowerBound for core::slice::SplitN<'a, T, P> where P: FnMut(&T) -> bool {}
    unsafe impl<'a, T, P> TrustedLowerBound for core::slice::SplitNMut<'a, T, P> where
        P: FnMut(&T) -> bool
    {
    }
    unsafe impl<T> TrustedLowerBound for core::slice::Windows<'_, T> {}
    unsafe impl TrustedLowerBound for core::str::Bytes<'_> {}
    unsafe impl TrustedLowerBound for core::str::EscapeDebug<'_> {}
    unsafe impl TrustedLowerBound for core::str::EscapeDefault<'_> {}
    unsafe impl TrustedLowerBound for core::str::EscapeUnicode<'_> {}
    unsafe impl TrustedLowerBound for core::str::Lines<'_> {}

    #[cfg(feature = "std")]
    mod std {
        use crate::TrustedLowerBound;
        use core::hash::{BuildHasher, Hash};
        use std::{
            env::{Args, ArgsOs, SplitPaths, Vars, VarsOs},
            fs::ReadDir,
            io::{BufRead, Read},
            path::{Ancestors, Components},
            process::{CommandArgs, CommandEnvs},
            sync::mpsc::TryIter,
            vec::Splice,
        };

        unsafe impl TrustedLowerBound for Ancestors<'_> {}
        unsafe impl TrustedLowerBound for Args {}
        unsafe impl TrustedLowerBound for ArgsOs {}
        unsafe impl<I> TrustedLowerBound for Box<I> where I: TrustedLowerBound {}
        unsafe impl TrustedLowerBound for CommandArgs<'_> {}
        unsafe impl TrustedLowerBound for CommandEnvs<'_> {}
        unsafe impl TrustedLowerBound for Components<'_> {}
        unsafe impl TrustedLowerBound for ReadDir {}
        unsafe impl<I> TrustedLowerBound for Splice<'_, I> where I: TrustedLowerBound {}
        unsafe impl TrustedLowerBound for SplitPaths<'_> {}
        unsafe impl<T> TrustedLowerBound for TryIter<'_, T> {}
        unsafe impl TrustedLowerBound for Vars {}
        unsafe impl TrustedLowerBound for VarsOs {}
        unsafe impl<T> TrustedLowerBound for std::collections::binary_heap::Drain<'_, T> {}
        unsafe impl<T> TrustedLowerBound for std::collections::binary_heap::Iter<'_, T> {}
        unsafe impl<T> TrustedLowerBound for std::collections::binary_heap::IntoIter<T> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::btree_map::IntoIter<K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::btree_map::IntoKeys<K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::btree_map::IntoValues<K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::btree_map::Iter<'_, K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::btree_map::IterMut<'_, K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::btree_map::Keys<'_, K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::btree_map::Range<'_, K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::btree_map::RangeMut<'_, K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::btree_map::Values<'_, K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::btree_map::ValuesMut<'_, K, V> {}
        unsafe impl<T> TrustedLowerBound for std::collections::btree_set::IntoIter<T> {}
        unsafe impl<T> TrustedLowerBound for std::collections::btree_set::Iter<'_, T> {}
        unsafe impl<T> TrustedLowerBound for std::collections::btree_set::Range<'_, T> {}
        unsafe impl<T: Ord> TrustedLowerBound for std::collections::btree_set::SymmetricDifference<'_, T> {}
        unsafe impl<T: Ord> TrustedLowerBound for std::collections::btree_set::Union<'_, T> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::hash_map::Drain<'_, K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::hash_map::IntoIter<K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::hash_map::IntoKeys<K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::hash_map::IntoValues<K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::hash_map::Iter<'_, K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::hash_map::IterMut<'_, K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::hash_map::Keys<'_, K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::hash_map::Values<'_, K, V> {}
        unsafe impl<K, V> TrustedLowerBound for std::collections::hash_map::ValuesMut<'_, K, V> {}
        unsafe impl<'a, T, S> TrustedLowerBound for std::collections::hash_set::Difference<'a, T, S>
        where
            T: Eq + Hash,
            S: BuildHasher,
        {
        }
        unsafe impl<'a, T, S> TrustedLowerBound for std::collections::hash_set::Intersection<'a, T, S>
        where
            T: Eq + Hash,
            S: BuildHasher,
        {
        }
        unsafe impl<'a, T, S> TrustedLowerBound
            for std::collections::hash_set::SymmetricDifference<'a, T, S>
        where
            T: Eq + Hash,
            S: BuildHasher,
        {
        }
        unsafe impl<'a, T, S> TrustedLowerBound for std::collections::hash_set::Union<'a, T, S>
        where
            T: Eq + Hash,
            S: BuildHasher,
        {
        }
        unsafe impl<K> TrustedLowerBound for std::collections::hash_set::Drain<'_, K> {}
        unsafe impl<K> TrustedLowerBound for std::collections::hash_set::IntoIter<K> {}
        unsafe impl<K> TrustedLowerBound for std::collections::hash_set::Iter<'_, K> {}
        unsafe impl<T> TrustedLowerBound for std::collections::linked_list::IntoIter<T> {}
        unsafe impl<T> TrustedLowerBound for std::collections::linked_list::Iter<'_, T> {}
        unsafe impl<T> TrustedLowerBound for std::collections::linked_list::IterMut<'_, T> {}
        unsafe impl<T> TrustedLowerBound for std::collections::vec_deque::Drain<'_, T> {}
        unsafe impl<T> TrustedLowerBound for std::collections::vec_deque::IntoIter<T> {}
        unsafe impl<T> TrustedLowerBound for std::collections::vec_deque::Iter<'_, T> {}
        unsafe impl<T> TrustedLowerBound for std::collections::vec_deque::IterMut<'_, T> {}
        unsafe impl<R: Read> TrustedLowerBound for std::io::Bytes<R> {}
        unsafe impl<B: BufRead> TrustedLowerBound for std::io::Lines<B> {}
        unsafe impl<B: BufRead> TrustedLowerBound for std::io::Split<B> {}
        unsafe impl TrustedLowerBound for std::string::Drain<'_> {}
        unsafe impl TrustedLowerBound for std::net::Incoming<'_> {}
        unsafe impl TrustedLowerBound for std::path::Iter<'_> {}
        unsafe impl<T> TrustedLowerBound for std::sync::mpsc::IntoIter<T> {}
        unsafe impl<T> TrustedLowerBound for std::sync::mpsc::Iter<'_, T> {}
        unsafe impl<T> TrustedLowerBound for std::vec::Drain<'_, T> {}
        unsafe impl<T> TrustedLowerBound for std::vec::IntoIter<T> {}

        #[cfg(target_os = "windows")]
        mod windows {
            use crate::TrustedLowerBound;
            use std::os::windows::ffi::EncodeWide;

            unsafe impl TrustedLowerBound for EncodeWide<'_> {}
        }
    }
}

/// Manual implementation of [`TrustedLowerBound`] for an iterator
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AssertLowerBoundOk<I: Iterator>(I);
//
impl<I: Iterator> AssertLowerBoundOk<I> {
    /// Assert that the lower size bound provided by an iterator's `size_hint()`
    /// method is correct.
    ///
    /// # Safety
    ///
    /// The lower size bound must indeed be correct.
    #[inline]
    pub unsafe fn new(inner: I) -> Self {
        Self(inner)
    }
}
//
impl<I: DoubleEndedIterator> DoubleEndedIterator for AssertLowerBoundOk<I> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth_back(n)
    }
}
//
impl<I: ExactSizeIterator> ExactSizeIterator for AssertLowerBoundOk<I> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}
//
impl<I: FusedIterator> FusedIterator for AssertLowerBoundOk<I> {}
//
impl<I: Iterator> Iterator for AssertLowerBoundOk<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn count(self) -> usize
    where
        I: Sized,
    {
        self.0.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item>
    where
        I: Sized,
    {
        self.0.last()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth(n)
    }
}
//
// # Safety
//
// Safety assertion is offloaded to the `new()` constructor
unsafe impl<I: Iterator> TrustedLowerBound for AssertLowerBoundOk<I> {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use static_assertions::assert_impl_all;

    assert_impl_all!(
        std::slice::Iter<'static, u32>: FusedIterator, TrustedLowerBound
    );

    proptest! {
        #[test]
        fn assert_lower_bound_basic(data: Vec<u8>) {
            let raw = data.iter();
            // SAFETY: The size_hint of Vec's iterator is trusted
            let iter = unsafe { AssertLowerBoundOk::new(raw.clone()) };
            assert_eq!(iter.size_hint(), raw.size_hint());
            assert_eq!(iter.len(), raw.len());
            assert_eq!(iter.clone().count(), raw.clone().count());
            assert_eq!(iter.clone().next(), raw.clone().next());
            assert_eq!(iter.clone().next_back(), raw.clone().next_back());
            assert_eq!(iter.clone().last(), raw.clone().last());
        }

        #[test]
        fn assert_lower_bound_strided(data: Vec<u8>, stride: usize) {
            let raw = data.iter();
            // SAFETY: The size_hint of Vec's iterator is trusted
            let iter = unsafe { AssertLowerBoundOk::new(raw.clone()) };
            assert_eq!(iter.clone().nth(stride), raw.clone().nth(stride));
            assert_eq!(iter.clone().nth_back(stride), raw.clone().nth_back(stride));
        }

        #[test]
        fn any(dataset: Vec<u8>, needle: u8) {
            let predicate = |&item| item == needle;
            let expected = dataset.iter().any(predicate);
            prop_assert_eq!(dataset.iter().any_ilp::<1>(predicate), expected);
            prop_assert_eq!(dataset.iter().any_ilp::<2>(predicate), expected);
            prop_assert_eq!(dataset.iter().any_ilp::<3>(predicate), expected);
        }

        #[test]
        fn all(dataset: Vec<u8>, needle: u8) {
            let predicate = |&item| item == needle;
            let expected = dataset.iter().all(predicate);
            prop_assert_eq!(dataset.iter().all_ilp::<1>(predicate), expected);
            prop_assert_eq!(dataset.iter().all_ilp::<2>(predicate), expected);
            prop_assert_eq!(dataset.iter().all_ilp::<3>(predicate), expected);
        }

        #[test]
        fn find(dataset: Vec<u8>, needle: u8) {
            let predicate = |item: &&u8| **item == needle;
            let expected = dataset.iter().find(predicate);
            prop_assert_eq!(dataset.iter().find_ilp::<1>(predicate), expected);
            prop_assert_eq!(dataset.iter().find_ilp::<2>(predicate), expected);
            prop_assert_eq!(dataset.iter().find_ilp::<3>(predicate), expected);
        }

        #[test]
        fn find_map(dataset: Vec<u8>, needle: u8) {
            let find_map = |item: &u8| (*item == needle).then_some(42);
            let expected = dataset.iter().find_map(find_map);
            prop_assert_eq!(dataset.iter().find_map_ilp::<1, _>(find_map), expected);
            prop_assert_eq!(dataset.iter().find_map_ilp::<2, _>(find_map), expected);
            prop_assert_eq!(dataset.iter().find_map_ilp::<3, _>(find_map), expected);
        }

        #[test]
        fn position(dataset: Vec<u8>, needle: u8) {
            let predicate = |item: &u8| *item == needle;
            let expected = dataset.iter().position(predicate);
            prop_assert_eq!(dataset.iter().position_ilp::<1>(predicate), expected);
            prop_assert_eq!(dataset.iter().position_ilp::<2>(predicate), expected);
            prop_assert_eq!(dataset.iter().position_ilp::<3>(predicate), expected);
        }

        #[test]
        fn rposition(dataset: Vec<u8>, needle: u8) {
            let predicate = |item: &u8| *item == needle;
            let expected = dataset.iter().rposition(predicate);
            prop_assert_eq!(dataset.iter().rposition_ilp::<1>(predicate), expected);
            prop_assert_eq!(dataset.iter().rposition_ilp::<2>(predicate), expected);
            prop_assert_eq!(dataset.iter().rposition_ilp::<3>(predicate), expected);
        }

        #[test]
        fn fold(dataset: Vec<u8>) {
            let zero = || 0;
            let accumulate = |a, &b| a + b as u64;
            let merge = |a, b| a + b;
            let expected = dataset.iter().fold(zero(), accumulate);
            prop_assert_eq!(
                dataset.iter().fold_ilp::<1, _>(zero, accumulate, merge),
                expected
            );
            prop_assert_eq!(
                dataset.iter().fold_ilp::<2, _>(zero, accumulate, merge),
                expected
            );
            prop_assert_eq!(
                dataset.iter().fold_ilp::<3, _>(zero, accumulate, merge),
                expected
            );
        }

        #[test]
        fn reduce(dataset: Vec<u64>) {
            let reduce = |a: u64, b| a.wrapping_add(b);
            let expected = dataset.iter().copied().reduce(reduce);
            prop_assert_eq!(dataset.iter().copied().reduce_ilp::<1>(reduce), expected);
            prop_assert_eq!(dataset.iter().copied().reduce_ilp::<2>(reduce), expected);
            prop_assert_eq!(dataset.iter().copied().reduce_ilp::<3>(reduce), expected);
        }

        #[test]
        fn sum(dataset: Vec<u8>) {
            let dataset = dataset.into_iter().map(|i| i as u64).collect::<Vec<_>>();
            let expected = dataset.iter().copied().sum::<u64>();
            prop_assert_eq!(dataset.iter().copied().sum_ilp::<1, u64>(), expected);
            prop_assert_eq!(dataset.iter().copied().sum_ilp::<2, u64>(), expected);
            prop_assert_eq!(dataset.iter().copied().sum_ilp::<3, u64>(), expected);
        }

        #[test]
        fn product(dataset: Vec<u8>) {
            let dataset = dataset
                .into_iter()
                .map(|i| (i as f64 / 256.0) + 0.5)
                .collect::<Vec<_>>();
            let expected = dataset.iter().copied().product::<f64>();
            let assert_close = |result: f64| Ok(prop_assert!((result - expected).abs() < 1e-6 * expected.abs()));
            assert_close(dataset.iter().copied().product_ilp::<1, f64>())?;
            assert_close(dataset.iter().copied().product_ilp::<2, f64>())?;
            assert_close(dataset.iter().copied().product_ilp::<3, f64>())?;
        }
    }
}
