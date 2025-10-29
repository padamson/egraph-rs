# Active Context: egraph-rs

## Current Work Focus

The project has reached a mature state with comprehensive functionality across multiple domains. Current development focuses on advanced layout algorithms and cross-language binding optimizations:

1. **Complete Algorithm Suite**

   - **Graph Algorithms**: Connected components, shortest path, triangulation, layering
   - **Layout Algorithms**: SGD variants (Full/Sparse/Distance-Adjusted/Omega/Kernel-SGD), MDS (Classical/Pivot), Stress Majorization, Kamada-Kawai
   - **Community Detection**: Louvain, Label Propagation, Spectral Clustering, InfoMap with unified trait interface
   - **Specialized Features**: Edge bundling, overlap removal, separation constraints, quality metrics

2. **Cross-Platform Language Bindings**

   - **Python Bindings**: Complete PyO3-based API with comprehensive coverage
   - **WebAssembly Bindings**: JavaScript-friendly interfaces with comprehensive test coverage
   - **API Compatibility**: Drop-in replacements maintaining backward compatibility

3. **Robust Architecture**

   - **Modular Crate Design**: 15+ specialized crates for different functionality
   - **Trait-Based Interfaces**: Consistent APIs across algorithms
   - **Multiple Geometric Spaces**: Euclidean (2D/nD), Spherical, Hyperbolic, Torus drawings
   - **Quality Metrics**: Comprehensive evaluation suite for layout assessment

4. **Performance & Reliability**
   - **Algorithm Optimizations**: Fixed infinite loops, improved convergence, enhanced performance
   - **Memory Efficiency**: Replaced external dependencies with built-in Rust collections
   - **Comprehensive Testing**: Unit tests, integration tests, cross-language validation
   - **Documentation**: Complete API documentation with examples and best practices

## Recent Changes

- **DiffusionKernel Random Access Interface Implementation (2025-10-29)**

  - **Complete Interface Redesign**: Implemented new DiffusionKernel structure providing efficient random access to individual elements of exp(-tL) matrix
  - **Core Design Philosophy**:
    - Random access pattern: Users query K[i,j] on demand rather than computing full matrix
    - Memory efficiency: O(n × num_vectors) vs O(n²) for full matrix
    - User-controlled caching: Distance computation left to users for flexible strategies
    - Symmetry guarantee: K[i,j] == K[j,i] numerically guaranteed
  - **Rust Implementation** (`crates/layout/kernel-sgd/src/diffusion_kernel.rs`):
    - `DiffusionKernel<S>` struct with automatic lambda_max estimation or external specification
    - `new(graph, length, t, degree, num_vectors, rng)`: Constructor with power method
    - `new_with_lambda_max(...)`: Constructor accepting external lambda_max
    - `get(i, j)`: Random access to kernel matrix elements (O(num_vectors) complexity)
    - Internal architecture uses HutchinsonEstimator with Chebyshev approximation
  - **Python Bindings** (`crates/python/src/layout/sgd/diffusion_kernel.rs`):
    - Complete PyO3 wrapper with numpy-free implementation
    - Constructor and static method for lambda_max specification
    - Element access via `get(i, j)` returning f64
    - Edge weight callback support
  - **Test Coverage**: 25/25 tests passing (18 Rust + 7 Python)
  - **Files Created**: diffusion_kernel.rs (Rust), diffusion_kernel.rs (Python), test_diffusion_kernel.py
  - **Files Modified**: hutchinson.rs, kernel_sgd.rs, lib.rs, mod.rs

- **KernelSgd Python Bindings Implementation (2025-10-29)**

  - **Complete Python Wrapper**: Implemented comprehensive PyO3-based bindings for kernel-sgd layout algorithm
  - **Builder Pattern API**: Five configurable parameters with method chaining support
  - **Python Implementation** (`crates/python/src/layout/sgd/kernel_sgd.rs`):
    - `PyKernelSgd` class with fluent API
    - Parameters: t, num_vectors, degree, k, min_dist
    - `build(graph, length, rng)`: Automatic lambda_max estimation
    - `build_with_lambda_max(...)`: External lambda_max specification
  - **Test Suite**: 9 comprehensive test cases covering all functionality
  - **Integration Status**: Complete with zero warnings, all tests passing

- **Kernel-SGD Layout Algorithm Implementation (2025-10-29)**

  - **Complete Implementation**: New kernel-sgd crate using diffusion kernel exp(-tL) with Chebyshev approximation
  - **New Crates**: petgraph-linalg-spmv (sparse matrix ops), petgraph-layout-kernel-sgd
  - **Five-Module Architecture**:
    - power_method.rs: λ_max estimation via iterative refinement
    - chebyshev.rs: exp(-tL) approximation using Clenshaw's algorithm
    - hutchinson.rs: Symmetry-optimized trace estimation
    - diffusion_kernel.rs: Random access interface to kernel elements
    - kernel_sgd.rs: Builder pattern and SGD integration
  - **Configurable Parameters**: t (diffusion time), num_vectors, degree, k, min_dist
  - **Algorithm Complexity**: O(degree × (|V| + |E|) × num_vectors)
  - **Test Coverage**: 15/15 Rust tests passing
  - **Integration**: Complete with existing SGD framework

- **Recent Historical Improvements (2025)**

  - **Louvain Clustering Fix (Oct)**: Corrected modularity calculation for undirected graphs, fixed community selection logic
  - **Spectral Clustering Enhancement (Oct)**: Integrated linfa k-means for better cluster quality
  - **RdMds & Omega Separation (Oct)**: Separated spectral embedding from node pair generation
  - **Numpy Integration (Sep)**: Complete numpy support with PyO3 0.26 upgrade
  - **WeightedEdgeLength Algorithm (Sep)**: Degree-based edge weight calculation for SGD
  - **Python Documentation (Oct)**: Complete Getting Started and Tutorial sections with 100% doctest pass rate
  - **Makefile Task Runner (Oct)**: Standardized task execution across Rust and Python

## Next Steps

1. **Documentation Enhancement**

   - Tutorial content for advanced algorithms
   - Best practice guides for algorithm selection
   - Integration examples with popular frameworks

2. **Performance Benchmarking**

   - Systematic performance evaluation
   - Comparison with other graph libraries
   - Scalability analysis for large graphs

3. **Community Engagement**
   - Example applications and use cases
   - Integration guides for different frameworks
   - User feedback incorporation

## Active Decisions and Considerations

- **API Stability**: Maintaining backward compatibility while allowing improvements
- **Performance vs. Flexibility**: Balancing generic interfaces with performance requirements
- **Cross-Language Consistency**: Ensuring similar behavior across Rust, Python, and JavaScript
- **Memory Management**: Careful handling of large graphs, especially in WebAssembly context
- **Design Paradigms**: Choosing between object-oriented and functional approaches based on use case

## Important Patterns and Preferences

- **Trait-Based Design**: Unified interfaces for algorithm families (CommunityDetection, LayeringAlgorithm)
- **Builder Pattern**: Configurable construction of complex algorithms
- **Functional Programming**: Pure functions for stateless computations (eigenvalue algorithms)
- **Error Handling**: Explicit error handling with proper conversion across language boundaries
- **Modular Architecture**: Specialized crates for focused functionality
- **Testing Strategy**: Comprehensive coverage including cross-language validation

## Workflow Guidelines

### Commit Message Format

All commit messages must follow Conventional Commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**: feat, fix, docs, style, refactor, perf, test, chore

**Scoping Rules**:

- **Crate-specific changes**: Use crate name as scope (e.g., `petgraph-layout-kernel-sgd`)
- **Project-wide changes**: Omit scope (e.g., root configuration, memory-bank updates)

**Examples**:

- `feat(petgraph-layout-kernel-sgd): add DiffusionKernel random access interface`
- `fix(petgraph-clustering): correct Louvain modularity calculation`
- `docs: update memory bank with recent implementations`

### Final Confirmation Process

**This step must NEVER be skipped:**

1. Run all required checks:

   - `cargo fmt --all`
   - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
   - `cargo test --workspace` (or appropriate tests)

2. Prepare comprehensive summary:

   - List all files modified
   - Describe all changes made
   - Explain impact and benefits

3. Create proper commit message following Conventional Commits format

4. Present to user for approval with complete summary and proposed commit message

5. Only after user approval: Mark task as complete

## Current Development Patterns

Based on open files and recent work:

- **SGD Framework Evolution**: Unified concrete implementation replacing trait-based approach
- **Algorithm Variants**: Supporting Full, Sparse, Distance-Adjusted, Omega, and Kernel-SGD variants
- **Scheduler System**: Five different learning rate strategies with comprehensive integration
- **Cross-Language Integration**: Parallel development across Rust, Python, and WebAssembly
- **Testing Excellence**: Extensive test coverage ensuring behavioral consistency

## Learnings and Project Insights

- **Rust-First Design**: Starting with Rust ensures memory safety and performance
- **Language Binding Patterns**: PyO3 and wasm-bindgen provide excellent cross-language foundations
- **Algorithm Implementation**: Graph algorithms benefit from trait-based generic implementations
- **Performance Considerations**: External dependencies should be carefully evaluated
- **Testing Importance**: Cross-language testing reveals subtle implementation differences
- **Documentation Value**: Good documentation significantly improves adoption
- **Workflow Discipline**: Consistent commit conventions and confirmation processes are essential
- **Design Philosophy**: Data structures should be structs, stateless computations benefit from functional approaches
