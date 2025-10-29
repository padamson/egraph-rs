//! Sparse symmetric matrix representation and operations.

use ndarray::Array1;
use num_traits::Zero;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

/// A sparse symmetric matrix represented in edge list format.
///
/// This structure stores only the lower triangular part of the matrix
/// (where i < j) to avoid redundancy. For symmetric matrices, A[i,j] = A[j,i],
/// so we only need to store one of these values.
///
/// The matrix is represented as:
/// - A list of edges (i, j, value) where i < j (off-diagonal elements)
/// - A vector of diagonal elements
///
/// This representation is efficient for graph Laplacians and other sparse
/// symmetric matrices commonly used in spectral graph theory.
#[derive(Debug, Clone)]
pub struct SparseSymmetricMatrix<T> {
    /// Number of rows/columns in the matrix
    n: usize,
    /// Off-diagonal entries: (row, col, value) where row < col
    edges: Vec<(usize, usize, T)>,
    /// Diagonal entries: diagonal[i] is the (i,i) element
    diagonal: Vec<T>,
}

impl<T> SparseSymmetricMatrix<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + AddAssign + Default + Zero,
{
    /// Creates a new sparse symmetric matrix with specified dimension.
    ///
    /// # Parameters
    /// * `n` - The dimension of the matrix (n x n)
    ///
    /// # Returns
    /// A zero matrix of size n x n
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
            diagonal: vec![T::default(); n],
        }
    }

    /// Creates a sparse symmetric matrix from edges and diagonal elements.
    ///
    /// # Parameters
    /// * `n` - The dimension of the matrix
    /// * `edges` - Off-diagonal entries (i, j, value) where i < j
    /// * `diagonal` - Diagonal entries
    ///
    /// # Panics
    /// Panics if any edge has i >= j or if indices are out of bounds
    pub fn from_parts(n: usize, edges: Vec<(usize, usize, T)>, diagonal: Vec<T>) -> Self {
        // Validate edges
        for &(i, j, _) in &edges {
            assert!(i < j, "Edges must have i < j for lower triangular storage");
            assert!(i < n && j < n, "Edge indices out of bounds");
        }
        assert_eq!(diagonal.len(), n, "Diagonal length must match dimension");

        Self { n, edges, diagonal }
    }

    /// Returns the dimension of the matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Sets a diagonal element.
    pub fn set_diagonal(&mut self, i: usize, value: T) {
        assert!(i < self.n, "Index out of bounds");
        self.diagonal[i] = value;
    }

    /// Adds to a diagonal element.
    pub fn add_to_diagonal(&mut self, i: usize, value: T) {
        assert!(i < self.n, "Index out of bounds");
        self.diagonal[i] += value;
    }

    /// Adds an off-diagonal edge.
    ///
    /// # Parameters
    /// * `i` - Row index (must be < j)
    /// * `j` - Column index (must be > i)
    /// * `value` - The value to store
    ///
    /// # Panics
    /// Panics if i >= j or if indices are out of bounds
    pub fn add_edge(&mut self, i: usize, j: usize, value: T) {
        assert!(i < j, "i must be < j for lower triangular storage");
        assert!(i < self.n && j < self.n, "Indices out of bounds");
        self.edges.push((i, j, value));
    }

    /// Computes the matrix-vector product: y = A * x
    ///
    /// For a symmetric matrix, this leverages the symmetry:
    /// y[i] = diagonal[i] * x[i] + Σ(value * x[j] for all edges (i,j))
    ///                            + Σ(value * x[i] for all edges (j,i))
    ///
    /// Complexity: O(nnz) where nnz is the number of non-zero elements
    pub fn multiply(&self, x: &Array1<T>) -> Array1<T> {
        assert_eq!(x.len(), self.n, "Vector dimension mismatch");

        let mut y = Array1::zeros(self.n);

        // Add diagonal contribution: y[i] += diagonal[i] * x[i]
        for i in 0..self.n {
            y[i] = self.diagonal[i] * x[i];
        }

        // Add off-diagonal contributions using symmetry
        for &(i, j, value) in &self.edges {
            // A[i,j] * x[j] contributes to y[i]
            y[i] += value * x[j];
            // A[j,i] * x[i] contributes to y[j] (using symmetry A[j,i] = A[i,j])
            y[j] += value * x[i];
        }

        y
    }

    /// Computes the matrix-vector product in-place: y = A * x
    ///
    /// This is more memory-efficient than the allocating version.
    pub fn multiply_into(&self, x: &Array1<T>, y: &mut Array1<T>) {
        assert_eq!(x.len(), self.n, "Input vector dimension mismatch");
        assert_eq!(y.len(), self.n, "Output vector dimension mismatch");

        // Initialize with diagonal contribution
        for i in 0..self.n {
            y[i] = self.diagonal[i] * x[i];
        }

        // Add off-diagonal contributions using symmetry
        for &(i, j, value) in &self.edges {
            y[i] += value * x[j];
            y[j] += value * x[i];
        }
    }

    /// Returns a reference to the edges.
    pub fn edges(&self) -> &[(usize, usize, T)] {
        &self.edges
    }

    /// Returns a reference to the diagonal.
    pub fn diagonal(&self) -> &[T] {
        &self.diagonal
    }
}

impl<T> SparseSymmetricMatrix<T>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + Default
        + PartialOrd,
{
    /// Creates a scaled and shifted matrix: (scale * A) - shift * I
    ///
    /// This is useful for Chebyshev approximation where we need to scale
    /// the matrix to [-1, 1] range: A' = (2/λ_max) * A - I
    ///
    /// # Parameters
    /// * `scale` - Scaling factor for all matrix elements
    /// * `shift` - Value to subtract from diagonal elements
    pub fn scale_and_shift(&self, scale: T, shift: T) -> Self {
        let mut edges = Vec::with_capacity(self.edges.len());
        for &(i, j, value) in &self.edges {
            edges.push((i, j, scale * value));
        }

        let mut diagonal = Vec::with_capacity(self.n);
        for &d in &self.diagonal {
            diagonal.push(scale * d - shift);
        }

        Self {
            n: self.n,
            edges,
            diagonal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        assert_eq!(matrix.dim(), 3);
        assert_eq!(matrix.edges().len(), 0);
        assert_eq!(matrix.diagonal().len(), 3);
    }

    #[test]
    fn test_multiply_identity() {
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        matrix.set_diagonal(0, 1.0);
        matrix.set_diagonal(1, 1.0);
        matrix.set_diagonal(2, 1.0);

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = matrix.multiply(&x);

        assert_eq!(y[0], 1.0);
        assert_eq!(y[1], 2.0);
        assert_eq!(y[2], 3.0);
    }

    #[test]
    fn test_multiply_with_edges() {
        // Create a 3x3 symmetric matrix:
        // [2  1  0]
        // [1  2  1]
        // [0  1  2]
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        matrix.set_diagonal(0, 2.0);
        matrix.set_diagonal(1, 2.0);
        matrix.set_diagonal(2, 2.0);
        matrix.add_edge(0, 1, 1.0); // (0,1) and (1,0)
        matrix.add_edge(1, 2, 1.0); // (1,2) and (2,1)

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = matrix.multiply(&x);

        // y[0] = 2*1 + 1*2 = 4
        // y[1] = 1*1 + 2*2 + 1*3 = 8
        // y[2] = 1*2 + 2*3 = 8
        assert_eq!(y[0], 4.0);
        assert_eq!(y[1], 8.0);
        assert_eq!(y[2], 8.0);
    }

    #[test]
    fn test_multiply_into() {
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        matrix.set_diagonal(0, 2.0);
        matrix.set_diagonal(1, 2.0);
        matrix.set_diagonal(2, 2.0);
        matrix.add_edge(0, 1, 1.0);
        matrix.add_edge(1, 2, 1.0);

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut y = Array1::zeros(3);
        matrix.multiply_into(&x, &mut y);

        assert_eq!(y[0], 4.0);
        assert_eq!(y[1], 8.0);
        assert_eq!(y[2], 8.0);
    }

    #[test]
    fn test_scale_and_shift() {
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(2);
        matrix.set_diagonal(0, 2.0);
        matrix.set_diagonal(1, 2.0);
        matrix.add_edge(0, 1, 1.0);

        // Create (2*A) - I
        let scaled = matrix.scale_and_shift(2.0, 1.0);

        // Diagonal should be 2*2 - 1 = 3
        assert_eq!(scaled.diagonal()[0], 3.0);
        assert_eq!(scaled.diagonal()[1], 3.0);

        // Off-diagonal should be 2*1 = 2
        assert_eq!(scaled.edges()[0].2, 2.0);
    }

    #[test]
    #[should_panic(expected = "i must be < j")]
    fn test_add_edge_wrong_order() {
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        matrix.add_edge(1, 0, 1.0); // Should panic: i >= j
    }
}
