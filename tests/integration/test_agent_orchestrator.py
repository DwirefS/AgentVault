"""
Integration tests for Agent Orchestrator
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.agents.agent_orchestrator import (
    AgentOrchestrator,
    DeploymentConfig,
    DeploymentStrategy
)
from src.database.models import Agent


class TestAgentOrchestratorIntegration:
    """Integration tests for Agent Orchestrator"""
    
    @pytest.fixture
    def orchestrator_config(self):
        """Test orchestrator configuration"""
        return {
            "max_concurrent_deployments": 5,
            "deployment_timeout": 300,
            "health_check_interval": 30,
            "default_strategy": "rolling_update",
            "kubernetes_namespace": "test-namespace"
        }
    
    @pytest.fixture
    def orchestrator(self, orchestrator_config):
        """Create orchestrator instance"""
        return AgentOrchestrator(orchestrator_config)
    
    @pytest.fixture
    def sample_agent(self):
        """Create sample agent for testing"""
        return Agent(
            id="test-agent-123",
            name="Test Orchestration Agent",
            agent_type="langchain",
            configuration={
                "runtime_type": "kubernetes",
                "namespace": "test-namespace",
                "image": "agentvault/langchain:latest",
                "cpu_cores": 2,
                "memory_gb": 4,
                "replicas": 3
            }
        )
    
    @pytest.fixture
    def deployment_config(self):
        """Create deployment configuration"""
        return DeploymentConfig(
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            max_unavailable=1,
            max_surge=1,
            timeout=300,
            health_check_path="/health",
            environment_variables={"ENV": "test"}
        )
    
    @pytest.mark.asyncio
    async def test_rolling_update_deployment(self, orchestrator, sample_agent, deployment_config):
        """Test rolling update deployment strategy"""
        with patch('kubernetes.client.AppsV1Api') as mock_k8s:
            mock_api = Mock()
            mock_k8s.return_value = mock_api
            
            # Mock successful deployment
            mock_api.patch_namespaced_deployment.return_value = None
            mock_api.read_namespaced_deployment.return_value = Mock(
                status=Mock(ready_replicas=3, replicas=3)
            )
            
            result = await orchestrator.deploy_agent(sample_agent, deployment_config)
            
            assert result['success'] is True
            assert result['strategy'] == 'rolling_update'
            mock_api.patch_namespaced_deployment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_blue_green_deployment(self, orchestrator, sample_agent, deployment_config):
        """Test blue-green deployment strategy"""
        deployment_config.strategy = DeploymentStrategy.BLUE_GREEN
        
        with patch.object(orchestrator, '_get_deployment_environment') as mock_get_env:
            with patch.object(orchestrator, '_create_green_deployment') as mock_create:
                with patch.object(orchestrator, '_test_deployment') as mock_test:
                    with patch.object(orchestrator, '_switch_traffic') as mock_switch:
                        with patch.object(orchestrator, '_cleanup_deployment') as mock_cleanup:
                            
                            # Mock successful blue-green deployment
                            mock_get_env.return_value = {'id': 'blue-deployment'}
                            mock_create.return_value = {'id': 'green-deployment'}
                            mock_test.return_value = True
                            
                            result = await orchestrator.deploy_agent(sample_agent, deployment_config)
                            
                            assert result['success'] is True
                            assert result['strategy'] == 'blue_green'
                            mock_create.assert_called_once()
                            mock_switch.assert_called_once()
                            mock_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_canary_deployment(self, orchestrator, sample_agent, deployment_config):
        """Test canary deployment strategy"""
        deployment_config.strategy = DeploymentStrategy.CANARY
        
        with patch.object(orchestrator, '_create_canary_deployment') as mock_create_canary:
            with patch.object(orchestrator, '_configure_traffic_split') as mock_traffic:
                with patch.object(orchestrator, '_monitor_canary_health') as mock_monitor:
                    with patch.object(orchestrator, '_promote_canary') as mock_promote:
                        
                        # Mock successful canary deployment
                        mock_create_canary.return_value = {'id': 'canary-deployment'}
                        mock_monitor.return_value = True
                        
                        result = await orchestrator.deploy_agent(sample_agent, deployment_config)
                        
                        assert result['success'] is True
                        assert result['strategy'] == 'canary'
                        mock_create_canary.assert_called_once()
                        mock_traffic.assert_called()
                        mock_promote.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_deployment_failure_rollback(self, orchestrator, sample_agent, deployment_config):
        """Test deployment failure triggers rollback"""
        with patch.object(orchestrator, '_create_green_deployment') as mock_create:
            with patch.object(orchestrator, '_test_deployment') as mock_test:
                with patch.object(orchestrator, '_rollback_deployment') as mock_rollback:
                    
                    # Mock deployment failure
                    mock_create.return_value = {'id': 'failed-deployment'}
                    mock_test.return_value = False  # Deployment test fails
                    
                    result = await orchestrator.deploy_agent(sample_agent, deployment_config)
                    
                    assert result['success'] is False
                    mock_rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_deployments_limit(self, orchestrator, sample_agent, deployment_config):
        """Test concurrent deployment limit enforcement"""
        # Set low concurrent limit
        orchestrator.config['max_concurrent_deployments'] = 1
        
        async def slow_deploy(*args, **kwargs):
            await asyncio.sleep(1)
            return {'success': True, 'strategy': 'test'}
        
        with patch.object(orchestrator, '_execute_rolling_update', side_effect=slow_deploy):
            # Start multiple deployments concurrently
            tasks = []
            for i in range(3):
                agent_copy = Agent(
                    id=f"test-agent-{i}",
                    name=f"Test Agent {i}",
                    agent_type="langchain",
                    configuration=sample_agent.configuration.copy()
                )
                task = orchestrator.deploy_agent(agent_copy, deployment_config)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should handle concurrent limit gracefully
            successful_deployments = [r for r in results if isinstance(r, dict) and r.get('success')]
            assert len(successful_deployments) >= 1
    
    @pytest.mark.asyncio
    async def test_health_check_monitoring(self, orchestrator, sample_agent):
        """Test health check monitoring during deployment"""
        with patch.object(orchestrator, '_monitor_canary_health') as mock_health:
            # Mock health check failure then success
            mock_health.side_effect = [False, False, True]
            
            deployment = {'id': 'test-deployment'}
            result = await orchestrator._monitor_canary_health(sample_agent, deployment)
            
            # Should eventually succeed after retries
            assert result is True
            assert mock_health.call_count == 3
    
    @pytest.mark.asyncio
    async def test_traffic_splitting_configuration(self, orchestrator, sample_agent):
        """Test traffic splitting configuration"""
        weights = {'stable': 80.0, 'canary': 20.0}
        
        with patch.object(orchestrator, '_configure_istio_traffic_split') as mock_istio:
            sample_agent.configuration['runtime_type'] = 'kubernetes'
            
            await orchestrator._configure_traffic_split(sample_agent, weights)
            
            mock_istio.assert_called_once_with(sample_agent, weights)
    
    @pytest.mark.asyncio
    async def test_deployment_cleanup(self, orchestrator):
        """Test deployment cleanup functionality"""
        deployment = {
            'id': 'test-deployment',
            'type': 'kubernetes',
            'namespace': 'test-namespace',
            'agent_id': 'test-agent'
        }
        
        with patch('kubernetes.client.AppsV1Api') as mock_k8s:
            with patch('kubernetes.client.CoreV1Api') as mock_core:
                mock_apps = Mock()
                mock_v1 = Mock()
                mock_k8s.return_value = mock_apps
                mock_core.return_value = mock_v1
                
                await orchestrator._cleanup_deployment(deployment)
                
                # Should delete deployment and associated resources
                mock_apps.delete_namespaced_deployment.assert_called_once()
                mock_v1.delete_namespaced_service.assert_called_once()
                mock_v1.delete_namespaced_config_map.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_docker_deployment_support(self, orchestrator, sample_agent, deployment_config):
        """Test Docker deployment support"""
        sample_agent.configuration['runtime_type'] = 'docker'
        sample_agent.configuration['docker_image'] = 'test/agent:latest'
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = 'container-id-123'
            
            result = await orchestrator.deploy_agent(sample_agent, deployment_config)
            
            # Should handle Docker deployments
            assert mock_subprocess.called
    
    @pytest.mark.asyncio
    async def test_deployment_timeout_handling(self, orchestrator, sample_agent, deployment_config):
        """Test deployment timeout handling"""
        deployment_config.timeout = 1  # Very short timeout
        
        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return True
        
        with patch.object(orchestrator, '_wait_for_replicas_ready', side_effect=slow_operation):
            with pytest.raises(asyncio.TimeoutError):
                await orchestrator.deploy_agent(sample_agent, deployment_config)
    
    @pytest.mark.asyncio
    async def test_deployment_labeling(self, orchestrator):
        """Test deployment labeling functionality"""
        deployment = {
            'id': 'test-deployment',
            'type': 'kubernetes',
            'namespace': 'test-namespace'
        }
        
        with patch('kubernetes.client.AppsV1Api') as mock_k8s:
            mock_api = Mock()
            mock_k8s.return_value = mock_api
            
            await orchestrator._label_deployment(deployment, 'stable')
            
            mock_api.patch_namespaced_deployment.assert_called_once()
            # Verify label was applied
            call_args = mock_api.patch_namespaced_deployment.call_args
            assert 'stable' in str(call_args)


class TestAgentOrchestratorErrorHandling:
    """Test error handling in agent orchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for error testing"""
        return AgentOrchestrator({})
    
    @pytest.mark.asyncio
    async def test_kubernetes_api_failure(self, orchestrator):
        """Test handling of Kubernetes API failures"""
        agent = Agent(
            id="test-agent",
            name="Test Agent",
            agent_type="langchain",
            configuration={'runtime_type': 'kubernetes'}
        )
        
        deployment_config = DeploymentConfig(
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            timeout=30
        )
        
        with patch('kubernetes.client.AppsV1Api', side_effect=Exception("K8s API Error")):
            result = await orchestrator.deploy_agent(agent, deployment_config)
            
            assert result['success'] is False
            assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_docker_command_failure(self, orchestrator):
        """Test handling of Docker command failures"""
        agent = Agent(
            id="test-agent",
            name="Test Agent", 
            agent_type="langchain",
            configuration={'runtime_type': 'docker'}
        )
        
        deployment_config = DeploymentConfig(
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            timeout=30
        )
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 1
            mock_subprocess.return_value.stderr = "Docker error"
            
            result = await orchestrator.deploy_agent(agent, deployment_config)
            
            assert result['success'] is False
    
    @pytest.mark.asyncio
    async def test_canary_health_check_failure(self, orchestrator):
        """Test canary deployment health check failure"""
        agent = Agent(
            id="test-agent",
            name="Test Agent",
            agent_type="langchain",
            configuration={'runtime_type': 'kubernetes'}
        )
        
        deployment = {'id': 'canary-deployment'}
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock health check failure
            mock_session.return_value.__aenter__.return_value.get.side_effect = Exception("Health check failed")
            
            result = await orchestrator._monitor_canary_health(agent, deployment)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_traffic_split_configuration_failure(self, orchestrator):
        """Test traffic split configuration failure handling"""
        agent = Agent(
            id="test-agent",
            name="Test Agent",
            agent_type="langchain",
            configuration={'runtime_type': 'kubernetes'}
        )
        
        weights = {'stable': 80.0, 'canary': 20.0}
        
        with patch.object(orchestrator, '_configure_istio_traffic_split', side_effect=Exception("Istio config failed")):
            # Should not raise exception
            await orchestrator._configure_traffic_split(agent, weights)
    
    @pytest.mark.asyncio
    async def test_deployment_environment_not_found(self, orchestrator):
        """Test handling when deployment environment is not found"""
        agent = Agent(
            id="test-agent",
            name="Test Agent",
            agent_type="langchain",
            configuration={'runtime_type': 'kubernetes'}
        )
        
        # Should return None for non-existent environment
        result = await orchestrator._get_deployment_environment(agent, 'nonexistent')
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cleanup_failure_handling(self, orchestrator):
        """Test cleanup failure handling"""
        deployment = {
            'id': 'test-deployment',
            'type': 'kubernetes',
            'namespace': 'test-namespace'
        }
        
        with patch('kubernetes.client.AppsV1Api') as mock_k8s:
            mock_api = Mock()
            mock_k8s.return_value = mock_api
            mock_api.delete_namespaced_deployment.side_effect = Exception("Delete failed")
            
            # Should handle cleanup failures gracefully
            with pytest.raises(Exception):
                await orchestrator._cleanup_deployment(deployment)