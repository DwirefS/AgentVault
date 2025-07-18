#!/usr/bin/env python3
"""
AgentVault™ CLI - Command Line Interface
Manage and interact with AgentVault™ storage platform
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import click
import asyncio
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.panel import Panel
from rich import print as rprint

from ..utils.config_generator import ConfigGenerator
from ..utils.migrate import MigrationManager

console = Console()


@click.group()
@click.version_option(version='1.0.0', prog_name='AgentVault™')
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration file path')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, config, debug):
    """AgentVault™ - Enterprise AI Agent Storage Platform
    
    Revolutionizing AI agent storage with Azure NetApp Files.
    90% latency reduction, infinite scale, zero compromises.
    """
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['debug'] = debug
    
    if debug:
        console.print("[yellow]Debug mode enabled[/yellow]")


@cli.command()
@click.option('--output', '-o', type=click.Path(), 
              help='Output configuration file path')
@click.option('--template', '-t', type=click.Path(exists=True),
              help='Configuration template path')
@click.option('--validate-only', is_flag=True,
              help='Only validate configuration without saving')
@click.pass_context
def init(ctx, output, template, validate_only):
    """Initialize AgentVault™ configuration"""
    with console.status("[bold green]Initializing AgentVault™..."):
        generator = ConfigGenerator(template_path=template)
        
        if output:
            generator.output_path = output
        
        try:
            if validate_only:
                config = generator.generate_from_environment()
                if generator.validate_config(config):
                    console.print("[green]✓ Configuration is valid[/green]")
                else:
                    console.print("[red]✗ Configuration validation failed[/red]")
                    ctx.exit(1)
            else:
                generator.run()
                console.print("[green]✓ Configuration generated successfully[/green]")
                console.print(f"[dim]Configuration saved to: {generator.output_path}[/dim]")
        
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            ctx.exit(1)


@cli.group()
def agent():
    """Manage AI agents"""
    pass


@agent.command('register')
@click.option('--agent-id', '-i', required=True, help='Unique agent identifier')
@click.option('--name', '-n', required=True, help='Agent name')
@click.option('--framework', '-f', type=click.Choice(['langchain', 'autogen', 'crewai']),
              required=True, help='AI framework')
@click.option('--memory-size', '-m', type=int, default=10,
              help='Memory size in GB (default: 10)')
@click.option('--tier', '-t', type=click.Choice(['ultra', 'premium', 'standard']),
              default='premium', help='Performance tier')
@click.option('--api-url', default='http://localhost:8000',
              help='AgentVault API URL')
@click.pass_context
def register_agent(ctx, agent_id, name, framework, memory_size, tier, api_url):
    """Register a new AI agent"""
    
    agent_config = {
        "agent_id": agent_id,
        "name": name,
        "framework": framework,
        "memory_size_gb": memory_size,
        "performance_tier": tier
    }
    
    with console.status(f"[bold green]Registering agent '{name}'..."):
        try:
            response = httpx.post(f"{api_url}/agents/register", json=agent_config)
            response.raise_for_status()
            
            result = response.json()
            if result.get('success'):
                console.print(f"[green]✓ Agent '{name}' registered successfully![/green]")
                console.print(f"[dim]Agent ID: {agent_id}[/dim]")
                console.print(f"[dim]Volume ID: {result.get('volume_id')}[/dim]")
            else:
                console.print(f"[red]✗ Registration failed: {result.get('error')}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            ctx.exit(1)


@agent.command('list')
@click.option('--framework', '-f', type=click.Choice(['langchain', 'autogen', 'crewai']),
              help='Filter by framework')
@click.option('--api-url', default='http://localhost:8000',
              help='AgentVault API URL')
@click.pass_context
def list_agents(ctx, framework, api_url):
    """List registered agents"""
    
    with console.status("[bold green]Fetching agents..."):
        try:
            params = {}
            if framework:
                params['framework'] = framework
                
            response = httpx.get(f"{api_url}/agents", params=params)
            response.raise_for_status()
            
            agents = response.json()
            
            if not agents:
                console.print("[yellow]No agents found[/yellow]")
                return
            
            # Create table
            table = Table(title="Registered Agents")
            table.add_column("Agent ID", style="cyan")
            table.add_column("Name", style="magenta")
            table.add_column("Framework", style="green")
            table.add_column("Memory", style="yellow")
            table.add_column("Tier", style="blue")
            table.add_column("Status", style="red")
            
            for agent in agents:
                table.add_row(
                    agent['agent_id'],
                    agent['name'],
                    agent['framework'],
                    f"{agent['memory_size_gb']} GB",
                    agent['performance_tier'],
                    agent.get('status', 'active')
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            ctx.exit(1)


@agent.command('stats')
@click.argument('agent_id')
@click.option('--api-url', default='http://localhost:8000',
              help='AgentVault API URL')
@click.pass_context
def agent_stats(ctx, agent_id, api_url):
    """Get agent statistics"""
    
    with console.status(f"[bold green]Fetching stats for agent '{agent_id}'..."):
        try:
            response = httpx.get(f"{api_url}/agents/{agent_id}/stats")
            response.raise_for_status()
            
            stats = response.json()
            
            # Display stats in a panel
            stats_text = f"""
[bold]Agent Statistics[/bold]

Agent ID: [cyan]{stats['agent_id']}[/cyan]
Total Operations: [yellow]{stats['total_operations']:,}[/yellow]
Storage Used: [green]{stats['storage_used_gb']:.2f} GB[/green]
Cache Hit Rate: [blue]{stats['cache_hit_rate']:.1%}[/blue]
Average Latency: [red]{stats['avg_latency_ms']:.2f} ms[/red]

[bold]Operations by Type:[/bold]
  • Reads: {stats['operations']['read']:,}
  • Writes: {stats['operations']['write']:,}
  • Deletes: {stats['operations']['delete']:,}
  • Lists: {stats['operations']['list']:,}

[bold]Storage by Tier:[/bold]
  • Ultra: {stats['storage_by_tier'].get('ultra', 0):.2f} GB
  • Premium: {stats['storage_by_tier'].get('premium', 0):.2f} GB
  • Standard: {stats['storage_by_tier'].get('standard', 0):.2f} GB
"""
            
            panel = Panel(stats_text, title=f"Agent: {agent_id}", border_style="green")
            console.print(panel)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                console.print(f"[red]Agent '{agent_id}' not found[/red]")
            else:
                console.print(f"[red]Error: {str(e)}[/red]")
            ctx.exit(1)
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            ctx.exit(1)


@cli.group()
def storage():
    """Manage storage operations"""
    pass


@storage.command('write')
@click.option('--agent-id', '-a', required=True, help='Agent ID')
@click.option('--key', '-k', required=True, help='Storage key')
@click.option('--data', '-d', required=True, help='Data to store (JSON)')
@click.option('--api-url', default='http://localhost:8000',
              help='AgentVault API URL')
@click.pass_context
def storage_write(ctx, agent_id, key, data, api_url):
    """Write data to storage"""
    
    try:
        # Parse JSON data
        parsed_data = json.loads(data)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON data: {str(e)}[/red]")
        ctx.exit(1)
    
    request = {
        "agent_id": agent_id,
        "operation": "write",
        "key": key,
        "data": parsed_data
    }
    
    with console.status("[bold green]Writing data..."):
        try:
            response = httpx.post(f"{api_url}/storage/request", json=request)
            response.raise_for_status()
            
            result = response.json()
            if result.get('success'):
                console.print(f"[green]✓ Data written successfully[/green]")
                console.print(f"[dim]Location: {result.get('location')}[/dim]")
                console.print(f"[dim]Latency: {result['metrics']['latency_ms']:.2f} ms[/dim]")
            else:
                console.print(f"[red]✗ Write failed: {result.get('error')}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            ctx.exit(1)


@storage.command('read')
@click.option('--agent-id', '-a', required=True, help='Agent ID')
@click.option('--key', '-k', required=True, help='Storage key')
@click.option('--output', '-o', type=click.Choice(['json', 'yaml', 'pretty']),
              default='pretty', help='Output format')
@click.option('--api-url', default='http://localhost:8000',
              help='AgentVault API URL')
@click.pass_context
def storage_read(ctx, agent_id, key, output, api_url):
    """Read data from storage"""
    
    request = {
        "agent_id": agent_id,
        "operation": "read",
        "key": key
    }
    
    with console.status("[bold green]Reading data..."):
        try:
            response = httpx.post(f"{api_url}/storage/request", json=request)
            response.raise_for_status()
            
            result = response.json()
            if result.get('success'):
                data = result.get('data')
                
                if output == 'json':
                    console.print(json.dumps(data, indent=2))
                elif output == 'yaml':
                    console.print(yaml.dump(data, default_flow_style=False))
                else:  # pretty
                    syntax = Syntax(json.dumps(data, indent=2), "json", theme="monokai")
                    console.print(syntax)
                
                if result.get('cache_hit'):
                    console.print("[dim]→ Cache hit[/dim]")
                console.print(f"[dim]Latency: {result['metrics']['latency_ms']:.2f} ms[/dim]")
            else:
                console.print(f"[red]✗ Read failed: {result.get('error')}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            ctx.exit(1)


@cli.group()
def migrate():
    """Database migration commands"""
    pass


@migrate.command('run')
@click.option('--version', '-v', help='Target version')
@click.option('--migrations-dir', '-d', type=click.Path(exists=True),
              help='Migrations directory')
@click.pass_context
def run_migrations(ctx, version, migrations_dir):
    """Run pending migrations"""
    
    async def _run():
        manager = MigrationManager(migrations_dir=migrations_dir)
        
        with console.status("[bold green]Running migrations..."):
            results = await manager.run_migrations(target_version=version)
            
        if results['executed']:
            console.print(f"[green]✓ Executed {len(results['executed'])} migrations[/green]")
            for migration_id in results['executed']:
                console.print(f"  [dim]→ {migration_id}[/dim]")
        else:
            console.print("[yellow]No pending migrations[/yellow]")
            
        if results['failed']:
            console.print(f"[red]✗ {len(results['failed'])} migrations failed[/red]")
            for failure in results['failed']:
                console.print(f"  [red]→ {failure['id']}: {failure['error']}[/red]")
            ctx.exit(1)
    
    asyncio.run(_run())


@migrate.command('status')
@click.option('--migrations-dir', '-d', type=click.Path(exists=True),
              help='Migrations directory')
def migration_status(ctx, migrations_dir):
    """Show migration status"""
    
    manager = MigrationManager(migrations_dir=migrations_dir)
    status = manager.get_status()
    
    # Create status table
    table = Table(title="Migration Status")
    table.add_column("Migration ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Version", style="yellow")
    table.add_column("Status", style="green")
    
    for migration in status['migrations']:
        status_text = "[green]✓ Applied[/green]" if migration['applied'] else "[yellow]⧖ Pending[/yellow]"
        table.add_row(
            migration['id'],
            migration['name'],
            migration['version'],
            status_text
        )
    
    console.print(table)
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total: {status['total_migrations']}")
    console.print(f"  Applied: {status['applied_migrations']}")
    console.print(f"  Pending: {status['pending_migrations']}")
    
    if status['current_version']:
        console.print(f"  Current Version: {status['current_version']}")


@cli.command()
@click.option('--api-url', default='http://localhost:8000',
              help='AgentVault API URL')
def health(api_url):
    """Check system health"""
    
    with console.status("[bold green]Checking system health..."):
        try:
            # Check API health
            response = httpx.get(f"{api_url}/health", timeout=5.0)
            api_healthy = response.status_code == 200
            api_data = response.json() if api_healthy else {}
            
            # Check metrics endpoint
            try:
                metrics_response = httpx.get(f"{api_url}/metrics", timeout=5.0)
                metrics_healthy = metrics_response.status_code == 200
            except:
                metrics_healthy = False
            
            # Display health status
            health_text = f"""
[bold]System Health Check[/bold]

API Status: {"[green]✓ Healthy[/green]" if api_healthy else "[red]✗ Unhealthy[/red]"}
Metrics: {"[green]✓ Available[/green]" if metrics_healthy else "[red]✗ Unavailable[/red]"}
Version: {api_data.get('version', 'Unknown')}
Environment: {api_data.get('environment', 'Unknown')}

[bold]Services:[/bold]
  • Storage Orchestrator: {api_data.get('services', {}).get('orchestrator', '[yellow]Unknown[/yellow]')}
  • Redis Cache: {api_data.get('services', {}).get('redis', '[yellow]Unknown[/yellow]')}
  • ML Engine: {api_data.get('services', {}).get('ml_engine', '[yellow]Unknown[/yellow]')}
"""
            
            if api_healthy:
                panel = Panel(health_text, title="AgentVault™ Health", border_style="green")
            else:
                panel = Panel(health_text, title="AgentVault™ Health", border_style="red")
            
            console.print(panel)
            
        except Exception as e:
            console.print(f"[red]Error checking health: {str(e)}[/red]")
            ctx.exit(1)


def main():
    """Main entry point"""
    cli()


if __name__ == '__main__':
    main()