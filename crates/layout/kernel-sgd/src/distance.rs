//! Distance computation from kernel matrix elements.
//!
//! This module computes Euclidean distances from kernel (inner product) matrix elements
//! using the formula: d(i,j)² = K[i,i] + K[j,j] - 2*K[i,j]

use crate::hutchinson::HutchinsonEstimator;
use num_traits::Float;

/// Computes the Euclidean distance between nodes i and j from kernel matrix elements.
///
/// Given a kernel matrix K that represents an inner product matrix, the Euclidean
/// distance between points i and j can be computed as:
/// d(i,j)² = K[i,i] + K[j,j] - 2*K[i,j]
/// d(i,j) = sqrt(K[i,i] + K[j,j] - 2*K[i,j])
///
/// # Parameters
/// * `estimator` - Hutchinson estimator for kernel matrix queries
/// * `i` - First node index
/// * `j` - Second node index
///
/// # Returns
/// Euclidean distance between nodes i and j
pub fn compute_distance<T>(estimator: &HutchinsonEstimator<T>, i: usize, j: usize) -> T
where
    T: Float + std::iter::Sum,
{
    if i == j {
        return T::zero();
    }

    let k_ii = estimator.query(i, i);
    let k_jj = estimator.query(j, j);
    let k_ij = estimator.query(i, j);

    // d² = K[i,i] + K[j,j] - 2*K[i,j]
    let dist_squared = k_ii + k_jj - T::from(2.0).unwrap() * k_ij;

    // Ensure non-negative due to numerical errors
    if dist_squared <= T::zero() {
        T::zero()
    } else {
        dist_squared.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hutchinson::{generate_rademacher_vectors, HutchinsonEstimator};
    use ndarray::Array2;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_compute_distance_identity() {
        // For identity matrix K=I, distances should follow:
        // d(i,i) = 0
        // d(i,j) = sqrt(1 + 1 - 2*0) = sqrt(2) for i≠j
        let mut rng = StdRng::seed_from_u64(42);
        let n = 10;
        let num_vectors = 200; // More vectors for better accuracy

        let v = generate_rademacher_vectors(n, num_vectors, &mut rng);
        let kv = v.clone(); // K = I

        let estimator = HutchinsonEstimator::new(v, kv);

        // Distance to self should be 0
        for i in 0..n {
            let dist: f64 = compute_distance(&estimator, i, i);
            assert!(
                dist.abs() < 1e-10,
                "Distance to self should be 0, got {}",
                dist
            );
        }

        // Distance between different nodes should be ~sqrt(2) ≈ 1.414
        let expected = 2.0_f64.sqrt();
        for i in 0..3 {
            for j in (i + 1)..4 {
                let dist: f64 = compute_distance(&estimator, i, j);
                assert!(
                    (dist - expected).abs() < 0.3,
                    "Distance({},{}) = {}, expected ≈ {}",
                    i,
                    j,
                    dist,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_compute_distance_scaled_identity() {
        // For K = c*I, distances should be sqrt(2c - 2c*0) = sqrt(2c)
        let mut rng = StdRng::seed_from_u64(42);
        let n = 10;
        let num_vectors = 200;
        let scale = 2.5;

        let v = generate_rademacher_vectors(n, num_vectors, &mut rng);
        let kv = &v * scale; // K = 2.5*I

        let estimator = HutchinsonEstimator::new(v, kv);

        // Distance between different nodes should be ~sqrt(2*2.5) ≈ 2.236
        let expected = (2.0 * scale).sqrt();
        for i in 0..3 {
            for j in (i + 1)..4 {
                let dist: f64 = compute_distance(&estimator, i, j);
                assert!(
                    (dist - expected).abs() < 0.4,
                    "Distance({},{}) = {}, expected ≈ {}",
                    i,
                    j,
                    dist,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_compute_distance_symmetry() {
        // Distance should be symmetric: d(i,j) = d(j,i)
        let mut rng = StdRng::seed_from_u64(42);
        let n = 5;
        let num_vectors = 100;

        let v = generate_rademacher_vectors(n, num_vectors, &mut rng);
        let mut kv = Array2::zeros((n, num_vectors));
        for i in 0..n {
            for j in 0..num_vectors {
                kv[[i, j]] = v[[i, j]] * 0.9 + v[[(i + 1) % n, j]] * 0.1;
            }
        }

        let estimator = HutchinsonEstimator::new(v, kv);

        for i in 0..3 {
            for j in (i + 1)..4 {
                let d_ij: f64 = compute_distance(&estimator, i, j);
                let d_ji: f64 = compute_distance(&estimator, j, i);
                assert!(
                    (d_ij - d_ji).abs() < 1e-10,
                    "Distance not symmetric: d({},{})={} != d({},{})={}",
                    i,
                    j,
                    d_ij,
                    j,
                    i,
                    d_ji
                );
            }
        }
    }
}
