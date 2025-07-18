#!/usr/bin/env python3
"""
Comprehensive test runner for AgentVault
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import sys
import os
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime


class AgentVaultTestRunner:
    """Comprehensive test runner for AgentVault"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = {}
        
    def run_unit_tests(self, verbose=False):
        """Run unit tests"""
        print("🧪 Running Unit Tests...")
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "--tb=short",
            "-v" if verbose else "-q",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/unit",
            "--junit-xml=test-results/unit-results.xml"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        self.test_results['unit'] = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            print("✅ Unit tests passed!")
        else:
            print("❌ Unit tests failed!")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return result.returncode == 0
    
    def run_integration_tests(self, verbose=False):
        """Run integration tests"""
        print("🔗 Running Integration Tests...")
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/integration/",
            "--tb=short",
            "-v" if verbose else "-q",
            "--junit-xml=test-results/integration-results.xml"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        self.test_results['integration'] = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            print("✅ Integration tests passed!")
        else:
            print("❌ Integration tests failed!")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return result.returncode == 0
    
    def run_e2e_tests(self, verbose=False):
        """Run end-to-end tests"""
        print("🌐 Running End-to-End Tests...")
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/e2e/",
            "--tb=short",
            "-v" if verbose else "-q",
            "--junit-xml=test-results/e2e-results.xml"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        self.test_results['e2e'] = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            print("✅ End-to-end tests passed!")
        else:
            print("❌ End-to-end tests failed!")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return result.returncode == 0
    
    def run_security_tests(self, verbose=False):
        """Run security-specific tests"""
        print("🔒 Running Security Tests...")
        cmd = [
            sys.executable, "-m", "pytest",
            "-m", "security",
            "--tb=short",
            "-v" if verbose else "-q",
            "--junit-xml=test-results/security-results.xml"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        self.test_results['security'] = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            print("✅ Security tests passed!")
        else:
            print("❌ Security tests failed!")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return result.returncode == 0
    
    def run_performance_tests(self, verbose=False):
        """Run performance tests"""
        print("⚡ Running Performance Tests...")
        cmd = [
            sys.executable, "-m", "pytest",
            "-m", "performance",
            "--tb=short",
            "-v" if verbose else "-q",
            "--junit-xml=test-results/performance-results.xml"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        self.test_results['performance'] = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            print("✅ Performance tests passed!")
        else:
            print("❌ Performance tests failed!")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return result.returncode == 0
    
    def run_linting(self, verbose=False):
        """Run code linting"""
        print("🧹 Running Code Linting...")
        
        # Run flake8
        flake8_cmd = ["flake8", "src/", "--max-line-length=120", "--ignore=E203,W503"]
        flake8_result = subprocess.run(flake8_cmd, cwd=self.project_root, capture_output=True, text=True)
        
        # Run black check
        black_cmd = ["black", "--check", "--diff", "src/"]
        black_result = subprocess.run(black_cmd, cwd=self.project_root, capture_output=True, text=True)
        
        # Run isort check
        isort_cmd = ["isort", "--check-only", "--diff", "src/"]
        isort_result = subprocess.run(isort_cmd, cwd=self.project_root, capture_output=True, text=True)
        
        linting_passed = all([
            flake8_result.returncode == 0,
            black_result.returncode == 0,
            isort_result.returncode == 0
        ])
        
        self.test_results['linting'] = {
            'flake8': flake8_result.returncode == 0,
            'black': black_result.returncode == 0,
            'isort': isort_result.returncode == 0,
            'passed': linting_passed
        }
        
        if linting_passed:
            print("✅ Code linting passed!")
        else:
            print("❌ Code linting failed!")
            if verbose:
                if flake8_result.returncode != 0:
                    print("Flake8 issues:")
                    print(flake8_result.stdout)
                if black_result.returncode != 0:
                    print("Black formatting issues:")
                    print(black_result.stdout)
                if isort_result.returncode != 0:
                    print("Import sorting issues:")
                    print(isort_result.stdout)
        
        return linting_passed
    
    def run_type_checking(self, verbose=False):
        """Run type checking with mypy"""
        print("🔍 Running Type Checking...")
        cmd = ["mypy", "src/", "--ignore-missing-imports", "--no-strict-optional"]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        self.test_results['type_checking'] = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            print("✅ Type checking passed!")
        else:
            print("❌ Type checking failed!")
            if verbose:
                print(result.stdout)
                print(result.stderr)
        
        return result.returncode == 0
    
    def run_dependency_check(self, verbose=False):
        """Check for security vulnerabilities in dependencies"""
        print("🔍 Checking Dependencies for Security Issues...")
        cmd = ["safety", "check", "--json"]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ No security vulnerabilities found in dependencies!")
            self.test_results['security_check'] = {'passed': True, 'vulnerabilities': []}
        else:
            print("⚠️  Security vulnerabilities found in dependencies!")
            try:
                vulnerabilities = json.loads(result.stdout)
                self.test_results['security_check'] = {
                    'passed': False,
                    'vulnerabilities': vulnerabilities
                }
                if verbose:
                    for vuln in vulnerabilities:
                        print(f"  - {vuln.get('package_name')}: {vuln.get('advisory')}")
            except json.JSONDecodeError:
                self.test_results['security_check'] = {
                    'passed': False,
                    'error': result.stdout
                }
        
        return result.returncode == 0
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📊 Generating Test Report...")
        
        # Ensure test-results directory exists
        results_dir = self.project_root / "test-results"
        results_dir.mkdir(exist_ok=True)
        
        # Create summary report
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'project': 'AgentVault',
            'test_results': self.test_results,
            'summary': {
                'total_suites': len(self.test_results),
                'passed_suites': sum(1 for result in self.test_results.values() 
                                   if isinstance(result, dict) and result.get('returncode') == 0 or result.get('passed') is True),
                'failed_suites': sum(1 for result in self.test_results.values() 
                                   if isinstance(result, dict) and result.get('returncode') != 0 or result.get('passed') is False)
            }
        }
        
        # Write JSON report
        with open(results_dir / "test-summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Write HTML report
        html_report = self.generate_html_report(summary)
        with open(results_dir / "test-report.html", 'w') as f:
            f.write(html_report)
        
        print(f"📄 Test report saved to {results_dir}/test-report.html")
        
        return summary
    
    def generate_html_report(self, summary):
        """Generate HTML test report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AgentVault Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .test-suite {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .passed {{ background-color: #d4edda; border-color: #c3e6cb; }}
                .failed {{ background-color: #f8d7da; border-color: #f5c6cb; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 AgentVault Test Report</h1>
                <p>Generated: {summary['timestamp']}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">Total Suites: {summary['summary']['total_suites']}</div>
                <div class="metric">Passed: {summary['summary']['passed_suites']}</div>
                <div class="metric">Failed: {summary['summary']['failed_suites']}</div>
            </div>
            
            <div class="test-results">
                <h2>Test Results</h2>
        """
        
        for suite_name, result in summary['test_results'].items():
            status = "passed" if (isinstance(result, dict) and 
                                (result.get('returncode') == 0 or result.get('passed') is True)) else "failed"
            
            html += f"""
                <div class="test-suite {status}">
                    <h3>{suite_name.title()} Tests</h3>
                    <p>Status: {'✅ Passed' if status == 'passed' else '❌ Failed'}</p>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def run_all_tests(self, verbose=False, quick=False):
        """Run all test suites"""
        print("🚀 Starting AgentVault Test Suite")
        print("=" * 50)
        
        # Ensure test-results directory exists
        results_dir = self.project_root / "test-results"
        results_dir.mkdir(exist_ok=True)
        
        all_passed = True
        
        # Run tests in order
        test_suites = [
            ("Linting", self.run_linting),
            ("Type Checking", self.run_type_checking),
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
        ]
        
        if not quick:
            test_suites.extend([
                ("Security Tests", self.run_security_tests),
                ("Performance Tests", self.run_performance_tests),
                ("E2E Tests", self.run_e2e_tests),
                ("Dependency Check", self.run_dependency_check),
            ])
        
        for suite_name, test_func in test_suites:
            print(f"\n{suite_name}:")
            print("-" * 30)
            try:
                passed = test_func(verbose=verbose)
                all_passed = all_passed and passed
            except Exception as e:
                print(f"❌ Error running {suite_name}: {str(e)}")
                all_passed = False
        
        # Generate report
        summary = self.generate_test_report()
        
        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 All tests passed! AgentVault is ready for production.")
        else:
            print("❌ Some tests failed. Please review and fix issues.")
        
        return all_passed


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AgentVault Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Run only essential tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--e2e", action="store_true", help="Run only e2e tests")
    parser.add_argument("--security", action="store_true", help="Run only security tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--lint", action="store_true", help="Run only linting")
    parser.add_argument("--type-check", action="store_true", help="Run only type checking")
    
    args = parser.parse_args()
    
    runner = AgentVaultTestRunner()
    
    if args.unit:
        success = runner.run_unit_tests(args.verbose)
    elif args.integration:
        success = runner.run_integration_tests(args.verbose)
    elif args.e2e:
        success = runner.run_e2e_tests(args.verbose)
    elif args.security:
        success = runner.run_security_tests(args.verbose)
    elif args.performance:
        success = runner.run_performance_tests(args.verbose)
    elif args.lint:
        success = runner.run_linting(args.verbose)
    elif args.type_check:
        success = runner.run_type_checking(args.verbose)
    else:
        success = runner.run_all_tests(args.verbose, args.quick)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()