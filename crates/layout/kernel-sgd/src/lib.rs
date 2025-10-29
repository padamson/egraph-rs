//! Kernel-based SGD layout algorithm using diffusion kernel approximation.
//!
//! This crate implements a graph layout algorithm that uses the diffusion kernel
//! exp(-tL) to compute ideal distances between nodes, where L is the graph Laplacian.
//! The kernel is approximated using Chebyshev polynomials and element queries are
//! performed using the Hutchinson trace estimator with symmetry optimization.

mod chebyshev;
mod diffusion_kernel;
mod hutchinson;
mod kernel_sgd;
mod power_method;

pub use diffusion_kernel::DiffusionKernel;
pub use hutchinson::HutchinsonEstimator;
pub use kernel_sgd::KernelSgd;
