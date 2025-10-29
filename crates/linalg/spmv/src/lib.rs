//! Sparse symmetric matrix-vector product implementation.
//!
//! This crate provides efficient sparse matrix-vector multiplication (spmv)
//! for symmetric matrices, which is a core operation for many graph algorithms
//! including Chebyshev polynomial approximation.

mod sparse_symmetric_matrix;

pub use sparse_symmetric_matrix::SparseSymmetricMatrix;
