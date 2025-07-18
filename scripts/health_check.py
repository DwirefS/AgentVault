#!/usr/bin/env python3
"""
AgentVault™ Health Check Script
Comprehensive system health verification
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import httpx
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class HealthChecker:
    """Comprehensive health checker for AgentVault™"""
    
    def __init__(self, api_url: str, environment: str = "production"):
        self.api_url = api_url
        self.environment = environment
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        
    async def check_api_health(self) -> Tuple[bool, Dict[str, Any]]:
        """Check API service health"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.api_url}/health")
                
                if response.status_code == 200:
                    data = response.json()
                    return True, {
                        "status": "healthy",
                        "version": data.get("version"),
                        "uptime": data.get("uptime"),
                        "environment": data.get("environment")
                    }
                else:
                    return False, {"status": "unhealthy", "code": response.status_code}
                    
        except Exception as e:
            return False, {"status": "unreachable", "error": str(e)}
    
    async def check_storage_connectivity(self) -> Tuple[bool, Dict[str, Any]]:
        """Check storage connectivity"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Test write operation
                test_data = {
                    "agent_id": "health-check-agent",
                    "operation": "write",
                    "key": f"health-check/{datetime.utcnow().isoformat()}",
                    "data": {"test": "health-check", "timestamp": datetime.utcnow().isoformat()}
                }
                
                write_response = await client.post(
                    f"{self.api_url}/storage/request",
                    json=test_data
                )
                
                if write_response.status_code != 200:
                    return False, {"status": "write_failed", "code": write_response.status_code}
                
                write_result = write_response.json()
                if not write_result.get("success"):
                    return False, {"status": "write_failed", "error": write_result.get("error")}
                
                # Test read operation
                read_data = {
                    "agent_id": "health-check-agent",
                    "operation": "read",
                    "key": test_data["key"]
                }
                
                read_response = await client.post(
                    f"{self.api_url}/storage/request",
                    json=read_data
                )
                
                if read_response.status_code != 200:
                    return False, {"status": "read_failed", "code": read_response.status_code}
                
                read_result = read_response.json()
                if not read_result.get("success"):
                    return False, {"status": "read_failed", "error": read_result.get("error")}
                
                # Clean up test data
                delete_data = {
                    "agent_id": "health-check-agent",
                    "operation": "delete",
                    "key": test_data["key"]
                }
                
                await client.post(f"{self.api_url}/storage/request", json=delete_data)
                
                return True, {
                    "status": "healthy",
                    "write_latency_ms": write_result.get("metrics", {}).get("latency_ms"),
                    "read_latency_ms": read_result.get("metrics", {}).get("latency_ms"),
                    "cache_hit": read_result.get("cache_hit", False)
                }
                
        except Exception as e:
            return False, {"status": "error", "error": str(e)}
    
    async def check_metrics_endpoint(self) -> Tuple[bool, Dict[str, Any]]:
        """Check Prometheus metrics endpoint"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.api_url}/metrics")
                
                if response.status_code == 200:
                    metrics_text = response.text
                    
                    # Check for key metrics
                    key_metrics = [
                        "agentvault_requests_total",
                        "agentvault_request_duration_seconds",
                        "agentvault_active_agents",
                        "agentvault_storage_usage_bytes"
                    ]
                    
                    found_metrics = []
                    for metric in key_metrics:
                        if metric in metrics_text:
                            found_metrics.append(metric)
                    
                    return True, {
                        "status": "healthy",
                        "total_metrics": len(found_metrics),
                        "key_metrics_found": found_metrics
                    }
                else:
                    return False, {"status": "unhealthy", "code": response.status_code}
                    
        except Exception as e:
            return False, {"status": "unreachable", "error": str(e)}
    
    async def check_redis_connectivity(self) -> Tuple[bool, Dict[str, Any]]:
        """Check Redis cache connectivity through API"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get system stats which should include Redis info
                response = await client.get(f"{self.api_url}/system/stats")
                
                if response.status_code == 200:
                    stats = response.json()
                    redis_stats = stats.get("cache", {})
                    
                    if redis_stats.get("connected"):
                        return True, {
                            "status": "healthy",
                            "memory_used": redis_stats.get("memory_used_mb"),
                            "hit_rate": redis_stats.get("hit_rate"),
                            "total_keys": redis_stats.get("total_keys")
                        }
                    else:
                        return False, {"status": "disconnected"}
                else:
                    return False, {"status": "unknown"}
                    
        except Exception as e:
            return False, {"status": "error", "error": str(e)}
    
    async def check_agent_operations(self) -> Tuple[bool, Dict[str, Any]]:
        """Check agent registration and operations"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # List agents
                response = await client.get(f"{self.api_url}/agents")
                
                if response.status_code == 200:
                    agents = response.json()
                    
                    # Count agents by framework
                    framework_counts = {}
                    for agent in agents:
                        framework = agent.get("framework", "unknown")
                        framework_counts[framework] = framework_counts.get(framework, 0) + 1
                    
                    return True, {
                        "status": "healthy",
                        "total_agents": len(agents),
                        "by_framework": framework_counts
                    }
                else:
                    return False, {"status": "unhealthy", "code": response.status_code}
                    
        except Exception as e:
            return False, {"status": "error", "error": str(e)}
    
    async def check_performance_metrics(self) -> Tuple[bool, Dict[str, Any]]:
        """Check system performance metrics"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.api_url}/system/performance")
                
                if response.status_code == 200:
                    perf = response.json()
                    
                    # Check performance thresholds
                    warnings = []
                    if perf.get("avg_latency_ms", 0) > 100:
                        warnings.append("High average latency detected")
                    if perf.get("error_rate", 0) > 0.01:
                        warnings.append("Error rate above 1%")
                    if perf.get("cpu_usage", 0) > 80:
                        warnings.append("High CPU usage")
                    
                    self.warnings.extend(warnings)
                    
                    return True, {
                        "status": "healthy" if not warnings else "warning",
                        "avg_latency_ms": perf.get("avg_latency_ms"),
                        "error_rate": perf.get("error_rate"),
                        "cpu_usage": perf.get("cpu_usage"),
                        "warnings": warnings
                    }
                else:
                    return False, {"status": "unavailable"}
                    
        except Exception as e:
            return False, {"status": "error", "error": str(e)}
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        checks = [
            ("API Health", self.check_api_health),
            ("Storage Connectivity", self.check_storage_connectivity),
            ("Metrics Endpoint", self.check_metrics_endpoint),
            ("Redis Cache", self.check_redis_connectivity),
            ("Agent Operations", self.check_agent_operations),
            ("Performance Metrics", self.check_performance_metrics)
        ]
        
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for check_name, check_func in checks:
                task = progress.add_task(f"Checking {check_name}...", total=1)
                
                passed, details = await check_func()
                results[check_name] = {
                    "passed": passed,
                    "details": details
                }
                
                if passed:
                    self.checks_passed += 1
                else:
                    self.checks_failed += 1
                
                progress.update(task, completed=1)
        
        return results
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display health check results"""
        # Create results table
        table = Table(title="AgentVault™ Health Check Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details", style="dim")
        
        for check_name, result in results.items():
            status = "[green]✓ PASSED[/green]" if result["passed"] else "[red]✗ FAILED[/red]"
            
            # Format details
            details = result["details"]
            if details.get("status") == "healthy":
                detail_text = "Healthy"
                if "latency_ms" in details:
                    detail_text += f" (latency: {details['latency_ms']:.1f}ms)"
            elif details.get("error"):
                detail_text = f"Error: {details['error'][:50]}..."
            else:
                detail_text = details.get("status", "Unknown")
            
            table.add_row(check_name, status, detail_text)
        
        console.print(table)
        
        # Summary panel
        total_checks = self.checks_passed + self.checks_failed
        health_score = (self.checks_passed / total_checks * 100) if total_checks > 0 else 0
        
        if health_score == 100:
            summary_style = "green"
            summary_emoji = "✅"
            summary_text = "All systems operational"
        elif health_score >= 80:
            summary_style = "yellow"
            summary_emoji = "⚠️"
            summary_text = "System operational with warnings"
        else:
            summary_style = "red"
            summary_emoji = "❌"
            summary_text = "System degraded"
        
        summary = f"""
{summary_emoji} [bold]{summary_text}[/bold]

Health Score: [bold]{health_score:.0f}%[/bold]
Checks Passed: {self.checks_passed}/{total_checks}
Environment: {self.environment}
Timestamp: {datetime.utcnow().isoformat()}
"""
        
        if self.warnings:
            summary += "\n[yellow]Warnings:[/yellow]\n"
            for warning in self.warnings:
                summary += f"  • {warning}\n"
        
        panel = Panel(summary, title="Health Check Summary", border_style=summary_style)
        console.print(panel)
        
        return health_score == 100


async def main():
    """Main health check function"""
    parser = argparse.ArgumentParser(description="AgentVault™ Health Check")
    parser.add_argument("--api-url", default="http://localhost:8000",
                      help="AgentVault API URL")
    parser.add_argument("--environment", default="production",
                      help="Environment name")
    parser.add_argument("--deployment", help="Deployment type (blue/green)")
    
    args = parser.parse_args()
    
    console.print(f"[bold blue]AgentVault™ Health Check[/bold blue]")
    console.print(f"API URL: {args.api_url}")
    console.print(f"Environment: {args.environment}\n")
    
    checker = HealthChecker(args.api_url, args.environment)
    
    try:
        results = await checker.run_all_checks()
        all_passed = checker.display_results(results)
        
        # Exit with appropriate code
        sys.exit(0 if all_passed else 1)
        
    except Exception as e:
        console.print(f"[red]Fatal error during health check: {str(e)}[/red]")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())