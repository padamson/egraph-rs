//! Kernel-based SGD layout algorithm
//!
//! This module provides Python bindings for the KernelSgd algorithm,
//! which uses the diffusion kernel exp(-tL) to compute ideal distances
//! between nodes for SGD optimization.

use crate::{
    graph::{GraphType, PyGraphAdapter},
    layout::sgd::PySgd,
    FloatType,
};
use petgraph::visit::EdgeRef;
use petgraph_layout_kernel_sgd::KernelSgd;
use pyo3::prelude::*;

/// Python class for configuring the KernelSgd algorithm
///
/// KernelSgd uses the diffusion kernel exp(-tL) to compute ideal distances
/// between nodes, where L is the graph Laplacian. The kernel is approximated
/// using Chebyshev polynomials and element queries are performed using the
/// Hutchinson trace estimator with symmetry optimization.
///
/// :param t: Diffusion time parameter (default: 1000.0)
/// :type t: float
/// :param num_vectors: Number of Hutchinson random vectors, effective 2x due to symmetry (default: 50)
/// :type num_vectors: int
/// :param degree: Degree of Chebyshev polynomial approximation (default: 10)
/// :type degree: int
/// :param k: Number of random pairs per node (default: 30)
/// :type k: int
/// :param min_dist: Minimum distance between node pairs (default: 1e-3)
/// :type min_dist: float
#[pyclass]
#[pyo3(name = "KernelSgd")]
pub struct PyKernelSgd {
    kernel_sgd: KernelSgd<FloatType>,
}

#[pymethods]
impl PyKernelSgd {
    /// Creates a new KernelSgd with default parameters
    ///
    /// Default values:
    /// - t: 1000.0 (diffusion time)
    /// - num_vectors: 50 (Hutchinson vectors, effective 100 with symmetry)
    /// - degree: 10 (Chebyshev polynomial degree)
    /// - k: 30 (random pairs per node)
    /// - min_dist: 1e-3 (minimum distance)
    ///
    /// :return: A new KernelSgd instance
    /// :rtype: KernelSgd
    #[new]
    fn new() -> Self {
        PyKernelSgd {
            kernel_sgd: KernelSgd::new(),
        }
    }

    /// Sets the diffusion time parameter
    ///
    /// :param t: Diffusion time parameter
    /// :type t: float
    /// :return: Self for method chaining
    /// :rtype: KernelSgd
    fn t(mut slf: PyRefMut<Self>, t: FloatType) -> Py<Self> {
        slf.kernel_sgd.t(t);
        slf.into()
    }

    /// Sets the number of Hutchinson random vectors
    ///
    /// Note: Due to symmetry optimization, effective sample count is 2x this value.
    ///
    /// :param num_vectors: Number of Hutchinson random vectors
    /// :type num_vectors: int
    /// :return: Self for method chaining
    /// :rtype: KernelSgd
    fn num_vectors(mut slf: PyRefMut<Self>, num_vectors: usize) -> Py<Self> {
        slf.kernel_sgd.num_vectors(num_vectors);
        slf.into()
    }

    /// Sets the degree of Chebyshev polynomial approximation
    ///
    /// :param degree: Degree of Chebyshev polynomial approximation
    /// :type degree: int
    /// :return: Self for method chaining
    /// :rtype: KernelSgd
    fn degree(mut slf: PyRefMut<Self>, degree: usize) -> Py<Self> {
        slf.kernel_sgd.degree(degree);
        slf.into()
    }

    /// Sets the number of random pairs per node
    ///
    /// :param k: Number of random pairs per node
    /// :type k: int
    /// :return: Self for method chaining
    /// :rtype: KernelSgd
    fn k(mut slf: PyRefMut<Self>, k: usize) -> Py<Self> {
        slf.kernel_sgd.k(k);
        slf.into()
    }

    /// Sets the minimum distance between node pairs
    ///
    /// :param min_dist: Minimum distance between node pairs
    /// :type min_dist: float
    /// :return: Self for method chaining
    /// :rtype: KernelSgd
    fn min_dist(mut slf: PyRefMut<Self>, min_dist: FloatType) -> Py<Self> {
        slf.kernel_sgd.min_dist(min_dist);
        slf.into()
    }

    /// Builds an Sgd instance with automatic lambda_max estimation
    ///
    /// Uses power method to estimate the maximum eigenvalue of the Laplacian.
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param length: A function that maps edge indices to their lengths/weights
    /// :type length: callable
    /// :param rng: Random number generator
    /// :type rng: Rng
    /// :return: A new Sgd instance configured with kernel-based distances
    /// :rtype: Sgd
    /// :raises: ValueError if the graph type is not supported
    fn build(
        &self,
        graph: &PyGraphAdapter,
        length: Py<PyAny>,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PySgd> {
        let sgd = match graph.graph() {
            GraphType::Graph(native_graph) => {
                let length_fn = |edge: petgraph::graph::EdgeReference<Py<PyAny>>| -> FloatType {
                    Python::attach(|py| {
                        let result = length.call1(py, (edge.id().index(),));
                        match result {
                            Ok(value) => value.extract::<FloatType>(py).unwrap_or(1.0),
                            Err(_) => 1.0,
                        }
                    })
                };
                self.kernel_sgd
                    .build(native_graph, length_fn, rng.get_mut())
            }
            _ => panic!("unsupported graph type"),
        };

        Ok(PySgd::new_with_sgd(sgd))
    }

    /// Builds an Sgd instance with externally provided lambda_max
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param length: A function that maps edge indices to their lengths/weights
    /// :type length: callable
    /// :param lambda_max: Maximum eigenvalue of the Laplacian
    /// :type lambda_max: float
    /// :param rng: Random number generator
    /// :type rng: Rng
    /// :return: A new Sgd instance configured with kernel-based distances
    /// :rtype: Sgd
    /// :raises: ValueError if the graph type is not supported
    fn build_with_lambda_max(
        &self,
        graph: &PyGraphAdapter,
        length: Py<PyAny>,
        lambda_max: FloatType,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PySgd> {
        let sgd = match graph.graph() {
            GraphType::Graph(native_graph) => {
                let length_fn = |edge: petgraph::graph::EdgeReference<Py<PyAny>>| -> FloatType {
                    Python::attach(|py| {
                        let result = length.call1(py, (edge.id().index(),));
                        match result {
                            Ok(value) => value.extract::<FloatType>(py).unwrap_or(1.0),
                            Err(_) => 1.0,
                        }
                    })
                };
                self.kernel_sgd.build_with_lambda_max(
                    native_graph,
                    length_fn,
                    lambda_max,
                    rng.get_mut(),
                )
            }
            _ => panic!("unsupported graph type"),
        };

        Ok(PySgd::new_with_sgd(sgd))
    }
}
