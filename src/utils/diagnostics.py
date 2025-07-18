"""
AgentVault™ Diagnostics Collector
System diagnostics and troubleshooting utilities
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import os
import json
import psutil
import platform
import socket
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import traceback
import aiofiles
import subprocess

logger = logging.getLogger(__name__)


class DiagnosticsCollector:
    """Comprehensive diagnostics collection for AgentVault™"""
    
    def __init__(self, output_dir: str = "/tmp/agentvault-diagnostics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.utcnow()
        
    async def collect_all(self) -> Dict[str, Any]:
        """Collect all diagnostic information"""
        logger.info("Starting comprehensive diagnostics collection...")
        
        diagnostics = {
            "metadata": {
                "collection_started": self.start_time.isoformat(),
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": platform.python_version()
            },
            "system": await self.collect_system_info(),
            "resources": await self.collect_resource_usage(),
            "network": await self.collect_network_info(),
            "storage": await self.collect_storage_info(),
            "processes": await self.collect_process_info(),
            "logs": await self.collect_recent_logs(),
            "configuration": await self.collect_configuration(),
            "performance": await self.collect_performance_metrics(),
            "errors": await self.collect_error_analysis()
        }
        
        diagnostics["metadata"]["collection_completed"] = datetime.utcnow().isoformat()
        diagnostics["metadata"]["collection_duration_seconds"] = (
            datetime.utcnow() - self.start_time
        ).total_seconds()
        
        # Save diagnostics to file
        await self.save_diagnostics(diagnostics)
        
        return diagnostics
    
    async def collect_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        try:
            return {
                "os": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor()
                },
                "hardware": {
                    "cpu_count": psutil.cpu_count(),
                    "cpu_count_logical": psutil.cpu_count(logical=True),
                    "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                    "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
                },
                "environment": dict(os.environ),
                "python": {
                    "version": platform.python_version(),
                    "implementation": platform.python_implementation(),
                    "compiler": platform.python_compiler()
                }
            }
        except Exception as e:
            logger.error(f"Error collecting system info: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    async def collect_resource_usage(self) -> Dict[str, Any]:
        """Collect current resource usage"""
        try:
            # CPU usage over 5 seconds
            cpu_samples = []
            for _ in range(5):
                cpu_samples.append(psutil.cpu_percent(interval=1))
            
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "cpu": {
                    "usage_percent_avg": sum(cpu_samples) / len(cpu_samples),
                    "usage_percent_samples": cpu_samples,
                    "per_cpu_percent": psutil.cpu_percent(interval=1, percpu=True),
                    "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory.percent,
                    "swap_total_gb": swap.total / (1024**3),
                    "swap_used_gb": swap.used / (1024**3),
                    "swap_percent": swap.percent
                },
                "disk_io": {
                    disk.device: {
                        "read_count": counters.read_count,
                        "write_count": counters.write_count,
                        "read_mb": counters.read_bytes / (1024**2),
                        "write_mb": counters.write_bytes / (1024**2)
                    }
                    for disk, counters in psutil.disk_io_counters(perdisk=True).items()
                }
            }
        except Exception as e:
            logger.error(f"Error collecting resource usage: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    async def collect_network_info(self) -> Dict[str, Any]:
        """Collect network information"""
        try:
            interfaces = {}
            for interface, addrs in psutil.net_if_addrs().items():
                interfaces[interface] = {
                    "addresses": [
                        {
                            "family": addr.family.name,
                            "address": addr.address,
                            "netmask": addr.netmask,
                            "broadcast": addr.broadcast
                        }
                        for addr in addrs
                    ]
                }
            
            # Network statistics
            net_io = psutil.net_io_counters()
            
            return {
                "interfaces": interfaces,
                "statistics": {
                    "bytes_sent_mb": net_io.bytes_sent / (1024**2),
                    "bytes_recv_mb": net_io.bytes_recv / (1024**2),
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                    "errors_in": net_io.errin,
                    "errors_out": net_io.errout,
                    "drop_in": net_io.dropin,
                    "drop_out": net_io.dropout
                },
                "connections": {
                    "total": len(psutil.net_connections()),
                    "by_status": self._count_connections_by_status()
                }
            }
        except Exception as e:
            logger.error(f"Error collecting network info: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _count_connections_by_status(self) -> Dict[str, int]:
        """Count network connections by status"""
        status_count = {}
        try:
            for conn in psutil.net_connections():
                status = conn.status
                status_count[status] = status_count.get(status, 0) + 1
        except:
            pass
        return status_count
    
    async def collect_storage_info(self) -> Dict[str, Any]:
        """Collect storage information"""
        try:
            partitions = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    partitions.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total_gb": usage.total / (1024**3),
                        "used_gb": usage.used / (1024**3),
                        "free_gb": usage.free / (1024**3),
                        "percent": usage.percent
                    })
                except PermissionError:
                    partitions.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "error": "Permission denied"
                    })
            
            # Check ANF mount
            anf_mount = os.environ.get("ANF_MOUNT_PATH", "/mnt/agentvault")
            anf_status = "not_mounted"
            
            if os.path.exists(anf_mount):
                try:
                    usage = psutil.disk_usage(anf_mount)
                    anf_status = {
                        "mounted": True,
                        "total_gb": usage.total / (1024**3),
                        "used_gb": usage.used / (1024**3),
                        "free_gb": usage.free / (1024**3),
                        "percent": usage.percent
                    }
                except:
                    anf_status = "error"
            
            return {
                "partitions": partitions,
                "anf_mount": anf_status
            }
        except Exception as e:
            logger.error(f"Error collecting storage info: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    async def collect_process_info(self) -> Dict[str, Any]:
        """Collect process information"""
        try:
            agentvault_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 
                                           'memory_percent', 'create_time', 'status']):
                try:
                    info = proc.info
                    cmdline = ' '.join(info.get('cmdline', []))
                    
                    # Find AgentVault processes
                    if 'agentvault' in info['name'].lower() or 'agentvault' in cmdline.lower():
                        agentvault_processes.append({
                            'pid': info['pid'],
                            'name': info['name'],
                            'cmdline': cmdline[:200],  # Truncate long command lines
                            'cpu_percent': info['cpu_percent'],
                            'memory_percent': info['memory_percent'],
                            'status': info['status'],
                            'uptime_seconds': (datetime.now() - datetime.fromtimestamp(
                                info['create_time']
                            )).total_seconds()
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                "agentvault_processes": agentvault_processes,
                "process_count": len(list(psutil.process_iter())),
                "top_cpu_processes": self._get_top_processes('cpu_percent', 5),
                "top_memory_processes": self._get_top_processes('memory_percent', 5)
            }
        except Exception as e:
            logger.error(f"Error collecting process info: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _get_top_processes(self, sort_by: str, limit: int) -> List[Dict[str, Any]]:
        """Get top processes by CPU or memory usage"""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', sort_by]):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort and return top N
        processes.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
        return processes[:limit]
    
    async def collect_recent_logs(self) -> Dict[str, Any]:
        """Collect recent log entries"""
        try:
            logs = {}
            log_locations = [
                ("/var/log/agentvault/agentvault.log", "application"),
                ("/var/log/agentvault/error.log", "error"),
                ("/var/log/syslog", "system")
            ]
            
            for log_path, log_type in log_locations:
                if os.path.exists(log_path):
                    try:
                        # Get last 100 lines
                        result = subprocess.run(
                            ["tail", "-n", "100", log_path],
                            capture_output=True,
                            text=True
                        )
                        logs[log_type] = result.stdout.split('\n')
                    except:
                        logs[log_type] = f"Could not read {log_path}"
                else:
                    logs[log_type] = f"Log file not found: {log_path}"
            
            return logs
        except Exception as e:
            logger.error(f"Error collecting logs: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    async def collect_configuration(self) -> Dict[str, Any]:
        """Collect configuration information"""
        try:
            config_data = {}
            
            # Environment variables (filtered)
            safe_env_vars = {}
            for key, value in os.environ.items():
                if key.startswith(('AGENTVAULT_', 'ANF_', 'REDIS_')):
                    # Mask sensitive values
                    if any(sensitive in key.lower() for sensitive in 
                           ['password', 'secret', 'key', 'token']):
                        safe_env_vars[key] = '***MASKED***'
                    else:
                        safe_env_vars[key] = value
            
            config_data['environment_variables'] = safe_env_vars
            
            # Configuration files
            config_files = [
                "/app/configs/config.yaml",
                "/app/configs/config.json"
            ]
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    try:
                        async with aiofiles.open(config_file, 'r') as f:
                            content = await f.read()
                            # Basic masking of sensitive data
                            masked_content = self._mask_sensitive_data(content)
                            config_data[os.path.basename(config_file)] = masked_content
                    except:
                        config_data[os.path.basename(config_file)] = "Could not read"
            
            return config_data
        except Exception as e:
            logger.error(f"Error collecting configuration: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _mask_sensitive_data(self, content: str) -> str:
        """Mask sensitive data in configuration content"""
        import re
        
        # Patterns for sensitive data
        patterns = [
            (r'(password|secret|key|token)(\s*[:=]\s*)["\']?([^"\'\n]+)["\']?', 
             r'\1\2***MASKED***'),
            (r'(sk-[a-zA-Z0-9]+)', r'sk-***MASKED***')
        ]
        
        masked = content
        for pattern, replacement in patterns:
            masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)
        
        return masked
    
    async def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics"""
        try:
            # This would typically integrate with your metrics system
            # For now, we'll collect basic performance data
            
            return {
                "current_load": {
                    "cpu_1min": os.getloadavg()[0],
                    "cpu_5min": os.getloadavg()[1],
                    "cpu_15min": os.getloadavg()[2]
                },
                "io_stats": psutil.disk_io_counters()._asdict(),
                "context_switches": psutil.cpu_stats().ctx_switches,
                "interrupts": psutil.cpu_stats().interrupts
            }
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    async def collect_error_analysis(self) -> Dict[str, Any]:
        """Analyze recent errors and issues"""
        try:
            # This would typically analyze logs for patterns
            # For now, return a simple structure
            
            return {
                "recent_errors": [],
                "error_patterns": {},
                "recommendations": [
                    "Monitor disk space usage",
                    "Check Redis connection stability",
                    "Verify ANF mount availability"
                ]
            }
        except Exception as e:
            logger.error(f"Error collecting error analysis: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    async def save_diagnostics(self, diagnostics: Dict[str, Any]) -> str:
        """Save diagnostics to file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"agentvault_diagnostics_{timestamp}.json"
        
        async with aiofiles.open(filename, 'w') as f:
            await f.write(json.dumps(diagnostics, indent=2, default=str))
        
        logger.info(f"Diagnostics saved to: {filename}")
        return str(filename)
    
    async def create_diagnostic_bundle(self) -> str:
        """Create a compressed diagnostic bundle"""
        diagnostics = await self.collect_all()
        
        # Create tarball
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        bundle_name = f"agentvault_diagnostics_{timestamp}.tar.gz"
        bundle_path = self.output_dir / bundle_name
        
        import tarfile
        with tarfile.open(bundle_path, "w:gz") as tar:
            # Add diagnostics JSON
            diag_file = self.output_dir / f"diagnostics_{timestamp}.json"
            with open(diag_file, 'w') as f:
                json.dump(diagnostics, f, indent=2, default=str)
            tar.add(diag_file, arcname="diagnostics.json")
            
            # Add any other relevant files
            # tar.add("/var/log/agentvault", arcname="logs")
            
        logger.info(f"Diagnostic bundle created: {bundle_path}")
        return str(bundle_path)