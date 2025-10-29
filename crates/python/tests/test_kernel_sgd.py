"""
Test suite for KernelSgd Python bindings
"""

import unittest
import egraph as eg


class TestKernelSgd(unittest.TestCase):
    """Test cases for KernelSgd class"""

    def setUp(self):
        """Set up a simple test graph"""
        self.graph = eg.Graph()
        self.n0 = self.graph.add_node(0)
        self.n1 = self.graph.add_node(1)
        self.n2 = self.graph.add_node(2)
        self.graph.add_edge(self.n0, self.n1, None)
        self.graph.add_edge(self.n1, self.n2, None)
        self.rng = eg.Rng.seed_from(42)

    def test_kernel_sgd_default_parameters(self):
        """Test KernelSgd with default parameters"""
        kernel_sgd = eg.KernelSgd()
        sgd = kernel_sgd.build(self.graph, lambda i: 1.0, self.rng)

        # Should create an Sgd instance
        self.assertIsInstance(sgd, eg.Sgd)

    def test_kernel_sgd_custom_parameters(self):
        """Test KernelSgd with custom parameters"""
        kernel_sgd = eg.KernelSgd()
        kernel_sgd.t(500.0)
        kernel_sgd.num_vectors(100)
        kernel_sgd.degree(20)
        kernel_sgd.k(50)
        kernel_sgd.min_dist(1e-2)

        sgd = kernel_sgd.build(self.graph, lambda i: 1.0, self.rng)

        self.assertIsInstance(sgd, eg.Sgd)

    def test_kernel_sgd_method_chaining(self):
        """Test KernelSgd method chaining"""
        rng = eg.Rng.seed_from(42)
        sgd = (
            eg.KernelSgd()
            .t(1000.0)
            .num_vectors(50)
            .degree(10)
            .k(30)
            .min_dist(1e-3)
            .build(self.graph, lambda i: 1.0, rng)
        )

        self.assertIsInstance(sgd, eg.Sgd)

    def test_kernel_sgd_with_lambda_max(self):
        """Test KernelSgd with externally provided lambda_max"""
        kernel_sgd = eg.KernelSgd()
        lambda_max = 2.5
        sgd = kernel_sgd.build_with_lambda_max(
            self.graph, lambda i: 1.0, lambda_max, self.rng
        )

        self.assertIsInstance(sgd, eg.Sgd)

    def test_kernel_sgd_weighted_edges(self):
        """Test KernelSgd with weighted edges"""
        # Create a graph with weighted edges
        graph = eg.Graph()
        n0 = graph.add_node(0)
        n1 = graph.add_node(1)
        n2 = graph.add_node(2)
        e0 = graph.add_edge(n0, n1, None)
        e1 = graph.add_edge(n1, n2, None)

        # Define edge weights
        weights = {e0: 2.0, e1: 3.0}

        kernel_sgd = eg.KernelSgd()
        rng = eg.Rng.seed_from(42)
        sgd = kernel_sgd.build(graph, lambda i: weights.get(i, 1.0), rng)

        self.assertIsInstance(sgd, eg.Sgd)

    def test_kernel_sgd_complete_layout(self):
        """Test complete layout process with KernelSgd"""
        # Create KernelSgd instance
        kernel_sgd = eg.KernelSgd()
        rng = eg.Rng.seed_from(42)
        sgd = kernel_sgd.build(self.graph, lambda i: 1.0, rng)

        # Create drawing
        drawing = eg.DrawingEuclidean2d.initial_placement(self.graph)

        # Create scheduler
        scheduler = sgd.scheduler(10, 0.1)

        # Run layout
        def step(eta):
            sgd.shuffle(rng)
            sgd.apply(drawing, eta)

        scheduler.run(step)

        # Verify positions changed from initial
        positions_changed = False
        for i in range(self.graph.node_count()):
            x, y = drawing.x(i), drawing.y(i)
            if abs(x) > 1e-6 or abs(y) > 1e-6:
                positions_changed = True
                break

        self.assertTrue(positions_changed)

    def test_kernel_sgd_all_schedulers(self):
        """Test KernelSgd with all scheduler types"""
        schedulers = [
            ("constant", lambda sgd: sgd.scheduler_constant(10, 0.1)),
            ("linear", lambda sgd: sgd.scheduler_linear(10, 0.1)),
            ("quadratic", lambda sgd: sgd.scheduler_quadratic(10, 0.1)),
            ("exponential", lambda sgd: sgd.scheduler_exponential(10, 0.1)),
            ("reciprocal", lambda sgd: sgd.scheduler_reciprocal(10, 0.1)),
        ]

        for name, create_scheduler in schedulers:
            with self.subTest(scheduler=name):
                kernel_sgd = eg.KernelSgd()
                rng = eg.Rng.seed_from(42)
                sgd = kernel_sgd.build(self.graph, lambda i: 1.0, rng)
                drawing = eg.DrawingEuclidean2d.initial_placement(self.graph)
                scheduler = create_scheduler(sgd)

                def step(eta):
                    sgd.shuffle(rng)
                    sgd.apply(drawing, eta)

                scheduler.run(step)

                # Verify scheduler completed
                self.assertTrue(scheduler.is_finished())

    def test_kernel_sgd_diffusion_time_parameter(self):
        """Test different diffusion time parameters"""
        times = [100.0, 1000.0, 10000.0]

        for t in times:
            with self.subTest(t=t):
                kernel_sgd = eg.KernelSgd()
                kernel_sgd.t(t)
                rng = eg.Rng.seed_from(42)
                sgd = kernel_sgd.build(self.graph, lambda i: 1.0, rng)

                self.assertIsInstance(sgd, eg.Sgd)

    def test_kernel_sgd_polynomial_degree(self):
        """Test different Chebyshev polynomial degrees"""
        degrees = [5, 10, 20]

        for degree in degrees:
            with self.subTest(degree=degree):
                kernel_sgd = eg.KernelSgd()
                kernel_sgd.degree(degree)
                rng = eg.Rng.seed_from(42)
                sgd = kernel_sgd.build(self.graph, lambda i: 1.0, rng)

                self.assertIsInstance(sgd, eg.Sgd)


if __name__ == "__main__":
    unittest.main()
