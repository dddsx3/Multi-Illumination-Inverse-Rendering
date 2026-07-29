"""
spectral_solver_diagnostics.py
==============================

谱求解器综合诊断与验证框架
包含：功能验证、性能基准、兼容性检查、回归测试

Usage:
    python spectral_solver_diagnostics.py --full
    python spectral_solver_diagnostics.py --quick
    python spectral_solver_diagnostics.py --performance
"""

import numpy as np
import scipy.linalg as la
import time
import sys
import traceback
import json
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import warnings
import inspect
import importlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# 导入被测模块
try:
    import spectral_solver
    import matrices
    import eigen_solver
    import mode_filter
    import chebyshev

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Module import failed: {e}")
    MODULES_AVAILABLE = False


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class TestResult:
    """单个测试结果数据类"""
    name: str
    status: str  # 'PASS', 'FAIL', 'ERROR', 'SKIP'
    duration: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkMetrics:
    """性能基准指标"""
    operation: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    memory_mb: float
    iterations: int
    throughput: float  # ops/sec

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """完整验证报告"""
    overall_status: str
    test_results: List[TestResult]
    benchmarks: List[BenchmarkMetrics]
    compatibility: Dict[str, Any]
    regressions: List[Dict[str, Any]]
    summary: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def export_json(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, default=str)

    def export_html(self, filepath: str):
        """生成可视化HTML报告"""
        html = self._generate_html()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

    def _generate_html(self) -> str:
        """生成HTML报告内容"""
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Spectral Solver Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
                .status-pass { color: #27ae60; font-weight: bold; }
                .status-fail { color: #e74c3c; font-weight: bold; }
                .status-error { color: #e67e22; font-weight: bold; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background: #34495e; color: white; }
                tr:nth-child(even) { background: #f2f2f2; }
                .metric-card { display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; min-width: 200px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 Spectral Solver Validation Report</h1>
                <p>Generated: {timestamp}</p>
                <p>Overall Status: <span class="status-{status_lower}">{status}</span></p>
            </div>

            <h2>📊 Summary</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>Total Tests</h3>
                    <p>{total_tests}</p>
                </div>
                <div class="metric-card">
                    <h3>Passed</h3>
                    <p class="status-pass">{passed}</p>
                </div>
                <div class="metric-card">
                    <h3>Failed</h3>
                    <p class="status-fail">{failed}</p>
                </div>
                <div class="metric-card">
                    <h3>Errors</h3>
                    <p class="status-error">{errors}</p>
                </div>
            </div>

            <h2>🧪 Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Duration (s)</th>
                    <th>Message</th>
                </tr>
                {test_rows}
            </table>

            <h2>⚡ Performance Benchmarks</h2>
            <table>
                <tr>
                    <th>Operation</th>
                    <th>Mean Time (s)</th>
                    <th>Throughput (ops/s)</th>
                    <th>Memory (MB)</th>
                </tr>
                {benchmark_rows}
            </table>
        </body>
        </html>
        '''

        # 生成测试行
        test_rows = ""
        for tr in self.test_results:
            status_class = f"status-{tr.status.lower()}"
            test_rows += f"""
                <tr>
                    <td>{tr.name}</td>
                    <td class="{status_class}">{tr.status}</td>
                    <td>{tr.duration:.4f}</td>
                    <td>{tr.message}</td>
                </tr>
            """

        # 生成基准行
        bench_rows = ""
        for bm in self.benchmarks:
            bench_rows += f"""
                <tr>
                    <td>{bm.operation}</td>
                    <td>{bm.mean_time:.6f}</td>
                    <td>{bm.throughput:.2f}</td>
                    <td>{bm.memory_mb:.2f}</td>
                </tr>
            """

        # 统计
        passed = sum(1 for tr in self.test_results if tr.status == 'PASS')
        failed = sum(1 for tr in self.test_results if tr.status == 'FAIL')
        errors = sum(1 for tr in self.test_results if tr.status == 'ERROR')

        return html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            status=self.overall_status,
            status_lower=self.overall_status.lower(),
            total_tests=len(self.test_results),
            passed=passed,
            failed=failed,
            errors=errors,
            test_rows=test_rows,
            benchmark_rows=bench_rows
        )


# =============================================================================
# 核心验证类
# =============================================================================

class SpectralSolverValidator:
    """
    谱求解器综合验证器

    提供四级验证体系：
    1. 功能验证：单元测试级别的基础功能检查
    2. 性能基准：计算效率与可扩展性测试
    3. 兼容性：环境与依赖检查
    4. 回归测试：与参考解对比
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.benchmarks: List[BenchmarkMetrics] = []
        self.reference_data: Optional[Dict] = None

        # 标准测试参数
        self.standard_ice = {
            'c_l': 3500.0,
            'c_s': 1800.0,
            'density': 917.0
        }
        self.standard_water = {
            'c_w': 1500.0,
            'density': 1024.0
        }

        # 加载参考数据（如果存在）
        self._load_reference_data()

    def _load_reference_data(self):
        """加载回归测试的参考数据"""
        ref_file = Path(__file__).parent / "test_reference_data.pkl"
        if ref_file.exists():
            with open(ref_file, 'rb') as f:
                self.reference_data = pickle.load(f)

    def _log(self, message: str, level: str = "INFO"):
        """内部日志"""
        if self.verbose:
            print(f"[{level}] {message}")

    def _create_solver(self, config: Optional[Dict] = None) -> 'spectral_solver.SpectralSolver':
        """工厂方法：创建配置好的求解器"""
        if not MODULES_AVAILABLE:
            raise RuntimeError("Required modules not available")

        cfg = config or {}
        return spectral_solver.SpectralSolver(cfg)

    def run_all_tests(self, test_categories: Optional[List[str]] = None) -> ValidationReport:
        """
        运行完整测试套件

        Args:
            test_categories: 指定测试类别 ['functional', 'performance', 'compatibility', 'regression']
                            None表示运行所有
        """
        categories = test_categories or ['functional', 'performance', 'compatibility', 'regression']

        self._log("Starting comprehensive validation suite...")
        start_time = time.time()

        # 1. 功能验证
        if 'functional' in categories:
            self._run_functional_tests()

        # 2. 性能基准
        if 'performance' in categories:
            self._run_performance_benchmarks()

        # 3. 兼容性测试
        if 'compatibility' in categories:
            self._run_compatibility_checks()

        # 4. 回归测试
        if 'regression' in categories:
            self._run_regression_tests()

        total_time = time.time() - start_time

        # 生成报告
        overall_status = self._determine_overall_status()
        summary = self._generate_summary(total_time)

        report = ValidationReport(
            overall_status=overall_status,
            test_results=self.results,
            benchmarks=self.benchmarks,
            compatibility=self._get_compatibility_info(),
            regressions=[r for r in self.results if 'regression' in r.name.lower()],
            summary=summary,
            metadata={
                'timestamp': datetime.now().isoformat(),
                'python_version': sys.version,
                'numpy_version': np.__version__,
                'total_duration': total_time,
                'categories_tested': categories
            }
        )

        self._log(f"Validation complete. Status: {overall_status}")
        return report

    # =========================================================================
    # 1. 功能验证套件
    # =========================================================================

    def _run_functional_tests(self):
        """运行功能验证测试"""
        self._log("Running functional tests...")

        test_methods = [
            self._test_initialization,
            self._test_chebyshev_discretization,
            self._test_matrix_assembly,
            self._test_eigen_solver,
            self._test_mode_filter,
            self._test_single_frequency_solve,
            self._test_frequency_sweep,
            self._test_dispersion_extraction,
            self._test_mode_shapes,
            self._test_batch_processing,
            self._test_configuration,
            self._test_callbacks
        ]

        for test_method in test_methods:
            try:
                result = test_method()
                self.results.append(result)
            except Exception as e:
                self.results.append(TestResult(
                    name=test_method.__name__,
                    status='ERROR',
                    duration=0.0,
                    message=str(e),
                    traceback=traceback.format_exc()
                ))

    def _test_initialization(self) -> TestResult:
        """测试求解器初始化"""
        start = time.time()
        try:
            solver = self._create_solver({'N': 30, 'thickness': 2.0})

            assert solver._assembler is not None
            assert solver._eigen_solver is not None
            assert solver._mode_filter is not None

            return TestResult(
                name="Initialization",
                status="PASS",
                duration=time.time() - start,
                message="Solver initialized successfully"
            )
        except Exception as e:
            return TestResult(
                name="Initialization",
                status="FAIL",
                duration=time.time() - start,
                message=str(e),
                traceback=traceback.format_exc()
            )

    def _test_chebyshev_discretization(self) -> TestResult:
        """测试切比雪夫离散化"""
        start = time.time()
        try:
            # 测试不同阶数
            for N in [10, 20, 30]:
                nodes, D = chebyshev.chebyshev_diff_matrix_order(N, 1)

                # 验证节点范围 [-1, 1]
                assert np.all(nodes >= -1) and np.all(nodes <= 1)
                assert nodes[0] == 1.0 and nodes[-1] == -1.0

                # 验证微分矩阵性质：行和应为0（常数导数为0）
                row_sums = np.sum(D, axis=1)
                assert np.allclose(row_sums, 0, atol=1e-10)

            return TestResult(
                name="Chebyshev Discretization",
                status="PASS",
                duration=time.time() - start,
                message=f"Verified orders: [10, 20, 30]"
            )
        except Exception as e:
            return TestResult(
                name="Chebyshev Discretization",
                status="FAIL",
                duration=time.time() - start,
                message=str(e)
            )

    def _test_matrix_assembly(self) -> TestResult:
        """测试矩阵组装"""
        start = time.time()
        try:
            solver = self._create_solver({'N': 20})
            omega = 2 * np.pi * 1000  # 1000 Hz

            # 测试矩阵A的组装
            A, B = solver._assembler.assemble_matrices(
                omega, self.standard_ice, self.standard_water, k=1.0
            )

            assert A.shape == (2 * (20 + 1), 2 * (20 + 1))
            assert np.iscomplexobj(A)
            assert not np.any(np.isnan(A))
            assert not np.any(np.isinf(A))

            return TestResult(
                name="Matrix Assembly",
                status="PASS",
                duration=time.time() - start,
                message=f"Matrix shape: {A.shape}, dtype: {A.dtype}"
            )
        except Exception as e:
            return TestResult(
                name="Matrix Assembly",
                status="FAIL",
                duration=time.time() - start,
                message=str(e)
            )

    def _test_eigen_solver(self) -> TestResult:
        """测试特征值求解器"""
        start = time.time()
        try:
            esolver = eigen_solver.ComplexEigenSolver(tol=1e-10)

            # 创建标准测试矩阵（对称复矩阵）
            N = 50
            A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
            A = A + A.T.conj()  # 厄米特化
            B = np.eye(N) + 0.1 * np.random.rand(N, N)

            evals, evecs = esolver.solve(A, B, sort_by='real_desc')

            # 验证残差 ||Ax - λBx||
            residuals = []
            for i in range(min(5, len(evals))):
                lhs = A @ evecs[:, i]
                rhs = evals[i] * (B @ evecs[:, i])
                res = np.linalg.norm(lhs - rhs) / np.linalg.norm(lhs)
                residuals.append(res)

            max_res = max(residuals)
            status = "PASS" if max_res < 1e-8 else "FAIL"

            return TestResult(
                name="Eigen Solver",
                status=status,
                duration=time.time() - start,
                message=f"Max residual: {max_res:.2e}, Found {len(evals)} modes",
                details={'max_residual': max_res, 'num_modes': len(evals)}
            )
        except Exception as e:
            return TestResult(
                name="Eigen Solver",
                status="ERROR",
                duration=time.time() - start,
                message=str(e),
                traceback=traceback.format_exc()
            )

    def _test_mode_filter(self) -> TestResult:
        """测试模态筛选器"""
        start = time.time()
        try:
            mf = mode_filter.ModeFilter(
                c_min=1000, c_max=6000,
                core_region=(-0.5, 0.5),
                energy_threshold=0.5
            )

            # 创建模拟特征值和向量
            N = 40
            k_array = np.array([5000 + 5j, 3000 - 2j, 8000 + 1j, 2000 + 0.1j])
            eigenvectors = np.random.rand(N, 4) + 1j * np.random.rand(N, 4)
            omega = 2 * np.pi * 1000
            y_grid = np.linspace(-1, 1, N)

            result = mf.filter_physical_modes(k_array, eigenvectors, omega, y_grid)

            # 验证返回结构
            assert 'wave_numbers' in result
            assert 'mode_types' in result
            assert 'confidences' in result

            return TestResult(
                name="Mode Filter",
                status="PASS",
                duration=time.time() - start,
                message=f"Filtered {len(result['wave_numbers'])} physical modes from {len(k_array)}"
            )
        except Exception as e:
            return TestResult(
                name="Mode Filter",
                status="FAIL",
                duration=time.time() - start,
                message=str(e)
            )

    def _test_single_frequency_solve(self) -> TestResult:
        """测试单频点求解"""
        start = time.time()
        try:
            solver = self._create_solver({'N': 25})
            freq = 100.0  # Hz

            result = solver.solve_single(freq, self.standard_ice, self.standard_water)

            assert 'modes' in result
            assert 'frequency' in result
            assert result['frequency'] == freq

            # 验证模态结构
            if result['modes']:
                mode = result['modes'][0]
                required_keys = ['wavenumber', 'phase_velocity', 'mode_type']
                for key in required_keys:
                    assert key in mode, f"Missing key: {key}"

            return TestResult(
                name="Single Frequency Solve",
                status="PASS",
                duration=time.time() - start,
                message=f"Found {len(result['modes'])} modes at {freq}Hz"
            )
        except Exception as e:
            return TestResult(
                name="Single Frequency Solve",
                status="FAIL",
                duration=time.time() - start,
                message=str(e),
                traceback=traceback.format_exc()
            )

    def _test_frequency_sweep(self) -> TestResult:
        """测试频率扫描"""
        start = time.time()
        try:
            solver = self._create_solver({'N': 20})
            freqs = np.linspace(50, 500, 10)

            results = solver.solve_sweep(freqs, self.standard_ice, self.standard_water, parallel=False)

            assert len(results) == len(freqs)

            # 验证所有频率都有结果
            for i, res in enumerate(results):
                assert res['frequency'] == freqs[i]

            return TestResult(
                name="Frequency Sweep",
                status="PASS",
                duration=time.time() - start,
                message=f"Solved {len(freqs)} frequencies sequentially"
            )
        except Exception as e:
            return TestResult(
                name="Frequency Sweep",
                status="FAIL",
                duration=time.time() - start,
                message=str(e)
            )

    def _test_dispersion_extraction(self) -> TestResult:
        """测试频散曲线提取"""
        start = time.time()
        try:
            solver = self._create_solver({'N': 20})
            freqs = np.linspace(100, 1000, 5)

            # 先求解
            results = solver.solve_sweep(freqs, self.standard_ice, self.standard_water, parallel=False)

            # 提取频散曲线
            dispersion = solver.get_dispersion_curves(results)

            assert 'frequencies' in dispersion
            assert 'wavenumbers' in dispersion
            assert 'phase_velocities' in dispersion

            return TestResult(
                name="Dispersion Extraction",
                status="PASS",
                duration=time.time() - start,
                message=f"Extracted {len(dispersion['frequencies'])} dispersion points"
            )
        except Exception as e:
            return TestResult(
                name="Dispersion Extraction",
                status="FAIL",
                duration=time.time() - start,
                message=str(e)
            )

    def _test_mode_shapes(self) -> TestResult:
        """测试模态形状计算"""
        start = time.time()
        try:
            solver = self._create_solver({'N': 20})
            result = solver.solve_single(200.0, self.standard_ice, self.standard_water)

            if not result['modes']:
                return TestResult(
                    name="Mode Shapes",
                    status="SKIP",
                    duration=time.time() - start,
                    message="No modes found to test shape extraction"
                )

            shapes = solver.compute_mode_shapes(result, mode_index=0)

            assert 'coordinates' in shapes
            assert 'phi' in shapes
            assert 'psi' in shapes

            return TestResult(
                name="Mode Shapes",
                status="PASS",
                duration=time.time() - start,
                message=f"Extracted shapes with {len(shapes['coordinates'])} points"
            )
        except Exception as e:
            return TestResult(
                name="Mode Shapes",
                status="FAIL",
                duration=time.time() - start,
                message=str(e)
            )

    def _test_batch_processing(self) -> TestResult:
        """测试批量处理"""
        start = time.time()
        try:
            solver = self._create_solver({'N': 20})

            # 创建多个参数集
            param_sets = []
            for i in range(3):
                ice = self.standard_ice.copy()
                ice['thickness'] = 1.0 + i * 0.5
                param_sets.append({
                    'frequencies': [100.0, 200.0, 300.0],
                    'ice_params': ice,
                    'water_params': self.standard_water
                })

            batch_results = solver.solve_batch(param_sets, parallel=False)

            assert len(batch_results) == len(param_sets)

            return TestResult(
                name="Batch Processing",
                status="PASS",
                duration=time.time() - start,
                message=f"Processed {len(param_sets)} parameter sets"
            )
        except Exception as e:
            return TestResult(
                name="Batch Processing",
                status="FAIL",
                duration=time.time() - start,
                message=str(e)
            )

    def _test_configuration(self) -> TestResult:
        """测试动态配置"""
        start = time.time()
        try:
            solver = self._create_solver({'N': 20})

            # 修改配置
            solver.configure(N=30, thickness=2.0)

            assert solver.config.N == 30
            assert solver.config.thickness == 2.0

            # 验证重新初始化
            assert solver._assembler.N == 30

            return TestResult(
                name="Dynamic Configuration",
                status="PASS",
                duration=time.time() - start,
                message="Configuration updated and components reinitialized"
            )
        except Exception as e:
            return TestResult(
                name="Dynamic Configuration",
                status="FAIL",
                duration=time.time() - start,
                message=str(e)
            )

    def _test_callbacks(self) -> TestResult:
        """测试回调系统"""
        start = time.time()
        try:
            solver = self._create_solver({'N': 15})

            events_triggered = []

            def progress_callback(data):
                events_triggered.append('progress')

            def complete_callback(data):
                events_triggered.append('complete')

            solver.register_callback('on_progress', progress_callback)
            solver.register_callback('on_complete', complete_callback)

            # 运行求解触发回调
            solver.solve_single(100.0, self.standard_ice, self.standard_water)

            assert 'progress' in events_triggered
            assert 'complete' in events_triggered

            return TestResult(
                name="Callback System",
                status="PASS",
                duration=time.time() - start,
                message=f"Triggered callbacks: {events_triggered}"
            )
        except Exception as e:
            return TestResult(
                name="Callback System",
                status="FAIL",
                duration=time.time() - start,
                message=str(e)
            )

    # =========================================================================
    # 2. 性能基准测试
    # =========================================================================

    def _run_performance_benchmarks(self):
        """运行性能基准测试"""
        self._log("Running performance benchmarks...")

        benchmarks = [
            self._benchmark_matrix_assembly,
            self._benchmark_eigen_decomposition,
            self._benchmark_frequency_sweep,
            self._benchmark_parallel_scaling,
            self._benchmark_memory_usage,
            self._benchmark_scalability
        ]

        for benchmark in benchmarks:
            try:
                metrics = benchmark()
                self.benchmarks.append(metrics)
            except Exception as e:
                self._log(f"Benchmark failed: {benchmark.__name__}: {e}", "ERROR")

    def _benchmark_operation(self, operation: Callable, iterations: int = 5,
                             warmup: int = 1) -> BenchmarkMetrics:
        """通用性能测试框架"""
        # Warmup
        for _ in range(warmup):
            operation()

        # 正式测试
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation()
            end = time.perf_counter()
            times.append(end - start)

        times = np.array(times)
        return BenchmarkMetrics(
            operation=operation.__name__,
            mean_time=float(np.mean(times)),
            std_time=float(np.std(times)),
            min_time=float(np.min(times)),
            max_time=float(np.max(times)),
            memory_mb=0.0,  # 可扩展内存监控
            iterations=iterations,
            throughput=iterations / float(np.sum(times))
        )

    def _benchmark_matrix_assembly(self) -> BenchmarkMetrics:
        """矩阵组装性能"""
        solver = self._create_solver({'N': 50})
        omega = 2 * np.pi * 1000

        def assemble():
            return solver._assembler.assemble_matrices(
                omega, self.standard_ice, self.standard_water, k=1.0
            )

        return self._benchmark_operation(assemble, iterations=10)

    def _benchmark_eigen_decomposition(self) -> BenchmarkMetrics:
        """特征分解性能"""
        N = 100
        A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        B = np.eye(N) + 0.01 * np.random.rand(N, N)
        esolver = eigen_solver.ComplexEigenSolver()

        def solve():
            return esolver.solve(A, B)

        return self._benchmark_operation(solve, iterations=5)

    def _benchmark_frequency_sweep(self) -> BenchmarkMetrics:
        """频率扫描性能"""
        solver = self._create_solver({'N': 30})
        freqs = np.linspace(100, 1000, 20)

        def sweep():
            return solver.solve_sweep(freqs, self.standard_ice, self.standard_water, parallel=False,
                                      show_progress=False)

        return self._benchmark_operation(sweep, iterations=3)

    def _benchmark_parallel_scaling(self) -> BenchmarkMetrics:
        """并行扩展性测试"""
        solver = self._create_solver({'N': 25})
        freqs = np.linspace(50, 500, 16)  # 16个频率点

        # 串行
        start = time.perf_counter()
        solver.solve_sweep(freqs, self.standard_ice, self.standard_water, parallel=False, show_progress=False)
        serial_time = time.perf_counter() - start

        # 并行
        start = time.perf_counter()
        solver.solve_sweep(freqs, self.standard_ice, self.standard_water, parallel=True, show_progress=False)
        parallel_time = time.perf_counter() - start

        speedup = serial_time / parallel_time if parallel_time > 0 else 1.0

        return BenchmarkMetrics(
            operation="Parallel Scaling (16 freq)",
            mean_time=parallel_time,
            std_time=0.0,
            min_time=parallel_time,
            max_time=serial_time,
            memory_mb=0.0,
            iterations=1,
            throughput=speedup
        )

    def _benchmark_memory_usage(self) -> BenchmarkMetrics:
        """内存使用基准"""
        # 简化版本，实际需要psutil
        import sys

        solver = self._create_solver({'N': 60})

        def create_large_matrix():
            # 创建大矩阵模拟内存使用
            A = np.random.rand(200, 200) + 1j * np.random.rand(200, 200)
            return A @ A.T

        start_mem = sys.getsizeof(locals())

        start = time.perf_counter()
        result = create_large_matrix()
        duration = time.perf_counter() - start

        end_mem = sys.getsizeof(locals())

        return BenchmarkMetrics(
            operation="Memory Allocation (N=200)",
            mean_time=duration,
            std_time=0.0,
            min_time=duration,
            max_time=duration,
            memory_mb=(end_mem - start_mem) / 1024 / 1024,
            iterations=1,
            throughput=1.0
        )

    def _benchmark_scalability(self) -> BenchmarkMetrics:
        """可扩展性测试（不同N的性能）"""
        N_values = [20, 40, 60, 80]
        times = []

        for N in N_values:
            solver = self._create_solver({'N': N})
            omega = 2 * np.pi * 500

            start = time.perf_counter()
            A, B = solver._assembler.assemble_matrices(
                omega, self.standard_ice, self.standard_water, k=1.0
            )
            duration = time.perf_counter() - start
            times.append(duration)

        # 计算复杂度拟合（应为O(N^3)左右）
        log_n = np.log(N_values)
        log_t = np.log(times)
        slope = np.polyfit(log_n, log_t, 1)[0]

        return BenchmarkMetrics(
            operation="Scalability (N=20-80)",
            mean_time=np.mean(times),
            std_time=np.std(times),
            min_time=np.min(times),
            max_time=np.max(times),
            memory_mb=0.0,
            iterations=len(N_values),
            throughput=slope  # 使用斜率表示复杂度
        )

    # =========================================================================
    # 3. 兼容性测试
    # =========================================================================

    def _run_compatibility_checks(self) -> Dict[str, Any]:
        """运行兼容性检查"""
        self._log("Running compatibility checks...")

        info = {
            'python_version': sys.version_info,
            'platform': sys.platform,
            'numpy_version': np.__version__,
            'scipy_version': self._get_scipy_version(),
            'multiprocessing_available': mp.cpu_count() > 1,
            'modules': {}
        }

        # 检查各模块接口兼容性
        modules_to_check = [
            ('spectral_solver', ['SpectralSolver', 'SolverConfig']),
            ('matrices', ['WaveguideMatrixAssembler']),
            ('eigen_solver', ['ComplexEigenSolver']),
            ('mode_filter', ['ModeFilter']),
            ('chebyshev', ['chebyshev_nodes', 'chebyshev_diff_matrix'])
        ]

        for module_name, expected_attrs in modules_to_check:
            try:
                mod = globals().get(module_name) or importlib.import_module(module_name)
                missing = []
                for attr in expected_attrs:
                    if not hasattr(mod, attr):
                        missing.append(attr)

                info['modules'][module_name] = {
                    'available': True,
                    'missing_attributes': missing,
                    'status': 'OK' if not missing else 'PARTIAL'
                }
            except Exception as e:
                info['modules'][module_name] = {
                    'available': False,
                    'error': str(e),
                    'status': 'FAIL'
                }

        # 数值精度检查
        info['numerical_precision'] = self._check_numerical_precision()

        return info

    def _get_scipy_version(self) -> str:
        """获取scipy版本"""
        try:
            import scipy
            return scipy.__version__
        except:
            return "unknown"

    def _check_numerical_precision(self) -> Dict[str, Any]:
        """检查数值精度"""
        tests = {}

        # 测试1：浮点精度
        eps = np.finfo(float).eps
        tests['machine_epsilon'] = float(eps)

        # 测试2：复杂运算精度
        A = np.random.rand(50, 50) + 1j * np.random.rand(50, 50)
        cond = np.linalg.cond(A)
        tests['matrix_condition_number'] = float(cond)
        tests['precision_warning'] = cond > 1e10

        return tests

    def _get_compatibility_info(self) -> Dict[str, Any]:
        """获取兼容性信息（供报告使用）"""
        return getattr(self, '_compat_info', {})

    # =========================================================================
    # 4. 回归测试
    # =========================================================================

    def _run_regression_tests(self):
        """运行回归测试"""
        self._log("Running regression tests...")

        # 如果不存在参考数据，生成新的参考数据
        if self.reference_data is None:
            self._log("No reference data found, generating new baseline...")
            self._generate_reference_data()
            self._log("Reference data generated. Run again to perform comparison.")
            return

        # 对比测试
        comparisons = [
            self._compare_eigenvalues,
            self._compare_dispersion_curves,
            self._compare_mode_shapes
        ]

        for compare_func in comparisons:
            try:
                result = compare_func()
                self.results.append(result)
            except Exception as e:
                self.results.append(TestResult(
                    name=f"Regression: {compare_func.__name__}",
                    status='ERROR',
                    duration=0.0,
                    message=str(e),
                    traceback=traceback.format_exc()
                ))

    def _generate_reference_data(self):
        """生成参考数据基线"""
        solver = self._create_solver({'N': 30})

        # 标准测试案例
        freqs = np.array([100.0, 500.0, 1000.0])

        data = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'solver_version': '1.0',
                'config': {'N': 30, 'thickness': 1.0}
            },
            'test_case_1': {}
        }

        for freq in freqs:
            result = solver.solve_single(freq, self.standard_ice, self.standard_water)
            data['test_case_1'][freq] = {
                'num_modes': len(result['modes']),
                'wavenumbers': [complex(m['wavenumber']) for m in result['modes'][:3]] if result['modes'] else [],
                'phase_velocities': [float(m['phase_velocity']) for m in result['modes'][:3]] if result['modes'] else []
            }

        # 保存参考数据
        ref_file = Path(__file__).parent / "test_reference_data.pkl"
        with open(ref_file, 'wb') as f:
            pickle.dump(data, f)

        self.reference_data = data

    def _compare_eigenvalues(self) -> TestResult:
        """对比特征值结果"""
        start = time.time()

        solver = self._create_solver(self.reference_data['metadata']['config'])
        freq = 500.0  # 选择中间频率

        result = solver.solve_single(freq, self.standard_ice, self.standard_water)
        ref = self.reference_data['test_case_1'][freq]

        # 对比前3个模态
        current_wn = np.array([complex(m['wavenumber']) for m in result['modes'][:3]])
        ref_wn = np.array(ref['wavenumbers'])

        if len(current_wn) != len(ref_wn):
            return TestResult(
                name="Regression: Eigenvalue Count",
                status="FAIL",
                duration=time.time() - start,
                message=f"Mode count mismatch: {len(current_wn)} vs {len(ref_wn)}"
            )

        # 计算相对误差
        errors = np.abs(current_wn - ref_wn) / np.abs(ref_wn)
        max_error = np.max(errors)

        status = "PASS" if max_error < 0.01 else "FAIL"  # 1%容差

        return TestResult(
            name="Regression: Eigenvalue Accuracy",
            status=status,
            duration=time.time() - start,
            message=f"Max relative error: {max_error:.2%}",
            details={'max_error': max_error, 'tolerance': 0.01}
        )

    def _compare_dispersion_curves(self) -> TestResult:
        """对比频散曲线"""
        start = time.time()

        # 生成新的频散曲线
        solver = self._create_solver({'N': 25})
        freqs = np.linspace(100, 1000, 10)
        results = solver.solve_sweep(freqs, self.standard_ice, self.standard_water, parallel=False, show_progress=False)

        # 简单对比：检查点数范围是否合理
        total_points = sum(len(r['modes']) for r in results)

        # 从参考数据估算合理范围
        expected_total = sum(self.reference_data['test_case_1'][f]['num_modes']
                             for f in [100.0, 500.0, 1000.0])
        expected_avg = expected_total / 3

        # 当前结果应该接近（允许30%偏差）
        current_avg = total_points / len(freqs)
        deviation = abs(current_avg - expected_avg) / expected_avg

        status = "PASS" if deviation < 0.3 else "FAIL"

        return TestResult(
            name="Regression: Dispersion Curve",
            status=status,
            duration=time.time() - start,
            message=f"Average modes per freq: {current_avg:.1f} (expected ~{expected_avg:.1f})",
            details={'deviation': deviation}
        )

    def _compare_mode_shapes(self) -> TestResult:
        """对比模态形状（简化的正交性检查）"""
        start = time.time()

        solver = self._create_solver({'N': 30})
        result = solver.solve_single(500.0, self.standard_ice, self.standard_water)

        if len(result['modes']) < 2:
            return TestResult(
                name="Regression: Mode Shape",
                status="SKIP",
                duration=time.time() - start,
                message="Insufficient modes for comparison"
            )

        # 检查前两个模态的正交性
        v1 = result['modes'][0]['eigenvector']
        v2 = result['modes'][1]['eigenvector']

        overlap = np.abs(np.vdot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        status = "PASS" if overlap < 0.9 else "FAIL"  # 应该近似正交

        return TestResult(
            name="Regression: Mode Orthogonality",
            status=status,
            duration=time.time() - start,
            message=f"Mode overlap: {overlap:.6f} (should be < 0.9)",
            details={'overlap': float(overlap)}
        )

    # =========================================================================
    # 报告生成
    # =========================================================================

    def _determine_overall_status(self) -> str:
        """确定整体状态"""
        if any(r.status == 'ERROR' for r in self.results):
            return 'ERROR'
        if any(r.status == 'FAIL' for r in self.results):
            return 'FAIL'
        if all(r.status == 'PASS' for r in self.results):
            return 'PASS'
        return 'PARTIAL'

    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """生成摘要统计"""
        status_counts = {'PASS': 0, 'FAIL': 0, 'ERROR': 0, 'SKIP': 0}
        for r in self.results:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1

        return {
            'total_tests': len(self.results),
            'passed': status_counts['PASS'],
            'failed': status_counts['FAIL'],
            'errors': status_counts['ERROR'],
            'skipped': status_counts['SKIP'],
            'total_duration': total_time,
            'benchmark_count': len(self.benchmarks)
        }


# =============================================================================
# 配置驱动接口（修改部分）
# =============================================================================

@dataclass
class DiagnosticConfig:
    """诊断配置 - 直接修改此处即可自定义测试"""
    # 测试类别选择（功能性、性能、兼容性、回归）
    test_categories: List[str] = field(default_factory=lambda: [
        'functional',
        # 'performance',  # 取消注释以启用
        # 'regression',   # 取消注释以启用
    ])

    # 全局参数
    verbose: bool = True
    generate_baseline: bool = False  # 是否重新生成参考数据

    # 测试参数覆盖
    chebyshev_orders: List[int] = field(default_factory=lambda: [20, 30, 40])
    frequency_ranges: Dict[str, Tuple[float, float, int]] = field(default_factory=lambda: {
        'low_freq': (10.0, 100.0, 5),
        'mid_freq': (100.0, 1000.0, 10),
        # 'high_freq': (1000.0, 5000.0, 20),
    })

    # 材料参数变体（用于参数研究）
    ice_variants: List[Dict] = field(default_factory=lambda: [
        {'c_l': 3500.0, 'c_s': 1800.0, 'density': 917.0},  # 标准冰
        # {'c_l': 3800.0, 'c_s': 1900.0, 'density': 900.0},  # 极地冰
    ])

    # 导出设置
    export_json: Optional[str] = "diagnostics_report.json"
    export_html: Optional[str] = "diagnostics_report.html"

    # 并行设置
    parallel_tests: bool = True
    max_workers: int = 4


def run_diagnostics(config: Optional[DiagnosticConfig] = None) -> ValidationReport:
    """
    脚本化入口 - 在Python环境中直接调用

    Example:
        # 快速测试
        report = run_diagnostics()

        # 自定义配置
        cfg = DiagnosticConfig()
        cfg.test_categories = ['performance']
        cfg.chebyshev_orders = [50, 100]
        report = run_diagnostics(cfg)
    """
    cfg = DiagnosticConfig()

    # 重新生成基线（如果需要）
    if cfg.generate_baseline:
        ref_file = Path(__file__).parent / "test_reference_data.pkl"
        if ref_file.exists():
            ref_file.unlink()
        print("参考数据已清除，将重新生成...")

    # 运行验证
    validator = SpectralSolverValidator(verbose=cfg.verbose)

    # 如果使用自定义测试参数，注入到验证器中
    if cfg.ice_variants:
        validator.standard_ice = cfg.ice_variants[0]

    report = validator.run_all_tests(test_categories=cfg.test_categories)

    # 导出报告
    if cfg.export_json:
        report.export_json(cfg.export_json)
        print(f"JSON报告已保存: {cfg.export_json}")

    if cfg.export_html:
        report.export_html(cfg.export_html)
        print(f"HTML报告已保存: {cfg.export_html}")

    # 打印摘要
    print("\n" + "=" * 60)
    print(f"诊断完成 | 状态: {report.overall_status}")
    print(f"测试通过: {report.summary['passed']}/{report.summary['total_tests']}")
    print("=" * 60)

    return report


# 保留向后兼容的命令行支持（可选）
def main():
    """保留simple CLI用于快速运行默认配置"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='配置文件路径(JSON)')
    args = parser.parse_args()

    if args.config:
        # 从文件加载配置
        with open(args.config) as f:
            config_dict = json.load(f)
        config = DiagnosticConfig(**config_dict)
    else:
        # 使用代码中的默认配置
        config = DiagnosticConfig()

    report = run_diagnostics(config)
    sys.exit(0 if report.overall_status == 'PASS' else 1)


if __name__ == "__main__":
    # 检查核心模块是否可用
    if not MODULES_AVAILABLE:
        print("警告: 缺少核心模块 (spectral_solver, matrices, eigen_solver, mode_filter, chebyshev)")
        print("这些模块需要在项目根目录下实现，或者将ice_root/src/physics目录添加到Python路径中")
        print("\n当前操作:")
        print("1. 显示配置信息")
        print("2. 生成兼容性报告")
        print("\n请在实现核心模块后再次运行完整测试")
        print("=" * 60)
    
    # 首次运行：全量全局测试模式
    # 包括功能、性能、兼容性和回归测试
    config = DiagnosticConfig()
    config.test_categories = ['functional', 'performance', 'compatibility', 'regression']
    config.chebyshev_orders = [20, 50, 100]  # 测试不同网格密度
    config.ice_variants = [
        {'c_l': 3500, 'c_s': 1800, 'density': 917},
        {'c_l': 3800, 'c_s': 1900, 'density': 900},  # 添加变体
    ]
    config.generate_baseline = True  # 首次运行，生成参考数据
    config.verbose = True
    config.export_html = "full_global_report.html"
    config.export_json = "full_global_report.json"
    config.parallel_tests = True
    config.max_workers = 4

    print("配置信息:")
    print("测试类别:", config.test_categories)
    print("切比雪夫阶数:", config.chebyshev_orders)
    print("冰参数变体:", len(config.ice_variants))
    print("是否生成基线:", config.generate_baseline)
    print("导出报告:", config.export_html)
    print("=" * 60)

    try:
        print("开始运行测试...")
        report = run_diagnostics(config)
        
        print("\n" + "=" * 60)
        print("测试完成!")
        print("整体状态:", report.overall_status)
        print("测试结果:")
        for r in report.test_results:
            print(f"  - {r.name}: {r.status} ({r.duration:.2f}s)")
        print("性能基准:")
        for b in report.benchmarks:
            print(f"  - {b.operation}: {b.mean_time:.4f}s")
        print("=" * 60)
    except Exception as e:
        print(f"测试运行时出错: {e}")
        print("请确保所有核心模块都已正确实现")
        print("=" * 60)
