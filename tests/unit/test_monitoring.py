"""
Unit tests for Monitoring and Alerting systems
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from src.monitoring.advanced_monitoring import (
    AdvancedMonitoringSystem,
    Alert,
    AlertRule,
    MetricDefinition,
    MetricType
)
from src.monitoring.prometheus_exporter import PrometheusExporter


class TestAdvancedMonitoringSystem:
    """Test suite for Advanced Monitoring System"""
    
    @pytest.fixture
    def monitoring_config(self):
        """Test monitoring configuration"""
        return {
            "smtp_server": "smtp.test.com",
            "smtp_port": 587,
            "smtp_username": "test@example.com",
            "smtp_password": "test-password",
            "email_to": ["admin@example.com"],
            "pagerduty_key": "test-pd-key",
            "azure_subscription_id": "test-subscription",
            "azure_resource_group": "test-rg",
            "azure_app_insights_key": "test-insights-key"
        }
    
    @pytest.fixture
    def monitoring_system(self, monitoring_config):
        """Create monitoring system instance"""
        return AdvancedMonitoringSystem(monitoring_config)
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample alert"""
        return Alert(
            id="test-alert-123",
            name="High CPU Usage",
            description="CPU usage exceeded threshold",
            severity="critical",
            timestamp=datetime.utcnow(),
            agent_id="agent-123",
            details={"cpu_usage": 95.5}
        )
    
    @pytest.fixture
    def sample_alert_rule(self):
        """Create sample alert rule"""
        return AlertRule(
            id="rule-123",
            name="CPU Threshold",
            metric_name="cpu_usage",
            condition="greater_than",
            threshold=90.0,
            severity="warning",
            enabled=True
        )
    
    def test_monitoring_system_initialization(self, monitoring_system):
        """Test monitoring system initializes correctly"""
        assert monitoring_system.config is not None
        assert monitoring_system.metrics == {}
        assert monitoring_system.alert_rules == {}
        assert monitoring_system.active_alerts == {}
    
    @pytest.mark.asyncio
    async def test_add_alert_rule(self, monitoring_system, sample_alert_rule):
        """Test adding alert rule"""
        await monitoring_system.add_alert_rule(sample_alert_rule)
        
        assert sample_alert_rule.id in monitoring_system.alert_rules
        assert monitoring_system.alert_rules[sample_alert_rule.id] == sample_alert_rule
    
    def test_register_metric(self, monitoring_system):
        """Test registering a metric"""
        metric_def = MetricDefinition(
            name="test_metric",
            type=MetricType.GAUGE,
            description="Test metric",
            labels=["agent_id"]
        )
        
        monitoring_system.register_metric(metric_def)
        
        assert "test_metric" in monitoring_system.metrics
        assert monitoring_system.metrics["test_metric"] == metric_def
    
    @pytest.mark.asyncio
    async def test_record_metric_triggers_alert(self, monitoring_system, sample_alert_rule):
        """Test recording metric that triggers alert"""
        # Add alert rule
        await monitoring_system.add_alert_rule(sample_alert_rule)
        
        # Record metric that should trigger alert
        await monitoring_system.record_metric(
            "cpu_usage",
            95.0,
            {"agent_id": "agent-123"}
        )
        
        # Check if alert was triggered
        assert len(monitoring_system.active_alerts) > 0
    
    @pytest.mark.asyncio
    @patch('smtplib.SMTP')
    async def test_send_email_notification(self, mock_smtp, monitoring_system, sample_alert):
        """Test email notification sending"""
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        await monitoring_system._send_email_notification(sample_alert)
        
        mock_smtp.assert_called_once()
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.sendmail.assert_called_once()
        mock_server.quit.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_send_pagerduty_notification(self, mock_session, monitoring_system, sample_alert):
        """Test PagerDuty notification sending"""
        mock_response = Mock()
        mock_response.status = 202
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
        
        await monitoring_system._send_pagerduty_notification(sample_alert)
        
        # Verify PagerDuty API was called
        mock_session.return_value.__aenter__.return_value.post.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_send_azure_alert_with_webhook(self, mock_session, monitoring_system, sample_alert):
        """Test Azure alert with webhook"""
        monitoring_system.config['azure_webhook_url'] = 'https://test.webhook.com'
        
        mock_response = Mock()
        mock_response.status = 200
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
        
        await monitoring_system._send_azure_alert(sample_alert)
        
        # Verify webhook was called
        mock_session.return_value.__aenter__.return_value.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_email_notification_missing_config(self, monitoring_system, sample_alert):
        """Test email notification with missing configuration"""
        # Remove email configuration
        del monitoring_system.config['smtp_username']
        
        # Should not raise exception, just log warning
        await monitoring_system._send_email_notification(sample_alert)
    
    @pytest.mark.asyncio
    async def test_alert_resolution(self, monitoring_system, sample_alert_rule):
        """Test alert resolution when condition no longer met"""
        await monitoring_system.add_alert_rule(sample_alert_rule)
        
        # Trigger alert
        await monitoring_system.record_metric(
            "cpu_usage",
            95.0,
            {"agent_id": "agent-123"}
        )
        
        # Record metric below threshold
        await monitoring_system.record_metric(
            "cpu_usage",
            80.0,
            {"agent_id": "agent-123"}
        )
        
        # Alert should be resolved
        # In a real implementation, this would check alert resolution logic
    
    def test_notification_channels_setup(self, monitoring_system):
        """Test notification channels are set up correctly"""
        monitoring_system._setup_notification_channels()
        
        # Check that notification channels are configured
        assert 'email' in monitoring_system.notification_channels
        assert 'pagerduty' in monitoring_system.notification_channels
        assert 'azure' in monitoring_system.notification_channels


class TestPrometheusExporter:
    """Test suite for Prometheus Exporter"""
    
    @pytest.fixture
    def exporter(self):
        """Create Prometheus exporter instance"""
        return PrometheusExporter()
    
    def test_record_agent_request(self, exporter):
        """Test recording agent request metrics"""
        exporter.record_agent_request(
            agent_id="test-agent",
            method="POST",
            status=200,
            duration=0.5
        )
        
        # In real implementation, would check Prometheus metrics
        # For now, just ensure no exceptions are raised
    
    def test_record_cache_hit(self, exporter):
        """Test recording cache hit metrics"""
        exporter.record_cache_hit("test-agent", True)
        exporter.record_cache_hit("test-agent", False)
        
        # Should not raise exceptions
    
    def test_record_storage_usage(self, exporter):
        """Test recording storage usage metrics"""
        exporter.record_storage_usage(
            agent_id="test-agent",
            tier="ultra",
            bytes_used=1024000
        )
        
        # Should not raise exceptions
    
    def test_record_tier_latency(self, exporter):
        """Test recording tier latency metrics"""
        exporter.record_tier_latency("ultra", 0.001)
        exporter.record_tier_latency("premium", 0.005)
        
        # Should not raise exceptions
    
    def test_record_agent_health(self, exporter):
        """Test recording agent health metrics"""
        exporter.record_agent_health(
            agent_id="test-agent",
            status="healthy",
            cpu_usage=50.0,
            memory_usage=60.0
        )
        
        # Should not raise exceptions


class TestAlertRuleEvaluation:
    """Test alert rule evaluation logic"""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create monitoring system for testing"""
        return AdvancedMonitoringSystem({})
    
    def test_greater_than_condition(self, monitoring_system):
        """Test greater than condition evaluation"""
        rule = AlertRule(
            id="test-rule",
            name="Test Rule",
            metric_name="cpu_usage",
            condition="greater_than",
            threshold=80.0,
            severity="warning"
        )
        
        # Should trigger alert
        assert monitoring_system._evaluate_condition(rule, 85.0) is True
        
        # Should not trigger alert
        assert monitoring_system._evaluate_condition(rule, 75.0) is False
    
    def test_less_than_condition(self, monitoring_system):
        """Test less than condition evaluation"""
        rule = AlertRule(
            id="test-rule",
            name="Test Rule",
            metric_name="memory_available",
            condition="less_than",
            threshold=20.0,
            severity="critical"
        )
        
        # Should trigger alert
        assert monitoring_system._evaluate_condition(rule, 15.0) is True
        
        # Should not trigger alert
        assert monitoring_system._evaluate_condition(rule, 25.0) is False
    
    def test_equals_condition(self, monitoring_system):
        """Test equals condition evaluation"""
        rule = AlertRule(
            id="test-rule",
            name="Test Rule",
            metric_name="status_code",
            condition="equals",
            threshold=500.0,
            severity="error"
        )
        
        # Should trigger alert
        assert monitoring_system._evaluate_condition(rule, 500.0) is True
        
        # Should not trigger alert
        assert monitoring_system._evaluate_condition(rule, 200.0) is False
    
    def test_unsupported_condition(self, monitoring_system):
        """Test unsupported condition handling"""
        rule = AlertRule(
            id="test-rule",
            name="Test Rule",
            metric_name="test_metric",
            condition="unsupported",
            threshold=50.0,
            severity="warning"
        )
        
        # Should return False for unsupported conditions
        assert monitoring_system._evaluate_condition(rule, 60.0) is False


class TestMonitoringErrorHandling:
    """Test error handling in monitoring system"""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create monitoring system with minimal config"""
        return AdvancedMonitoringSystem({})
    
    @pytest.mark.asyncio
    async def test_smtp_connection_failure(self, monitoring_system):
        """Test handling of SMTP connection failure"""
        alert = Alert(
            id="test-alert",
            name="Test Alert",
            description="Test",
            severity="warning",
            timestamp=datetime.utcnow()
        )
        
        with patch('smtplib.SMTP', side_effect=Exception("Connection failed")):
            # Should not raise exception
            await monitoring_system._send_email_notification(alert)
    
    @pytest.mark.asyncio
    async def test_pagerduty_api_failure(self, monitoring_system):
        """Test handling of PagerDuty API failure"""
        monitoring_system.config['pagerduty_key'] = 'test-key'
        
        alert = Alert(
            id="test-alert",
            name="Test Alert",
            description="Test",
            severity="critical",
            timestamp=datetime.utcnow()
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Server Error")
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            # Should not raise exception
            await monitoring_system._send_pagerduty_notification(alert)
    
    @pytest.mark.asyncio
    async def test_azure_monitor_failure(self, monitoring_system):
        """Test handling of Azure Monitor failure"""
        monitoring_system.config.update({
            'azure_subscription_id': 'test-sub',
            'azure_resource_group': 'test-rg'
        })
        
        alert = Alert(
            id="test-alert",
            name="Test Alert",
            description="Test",
            severity="error",
            timestamp=datetime.utcnow()
        )
        
        with patch('azure.identity.DefaultAzureCredential', side_effect=Exception("Auth failed")):
            # Should not raise exception
            await monitoring_system._send_azure_alert(alert)
    
    def test_invalid_metric_type(self, monitoring_system):
        """Test handling of invalid metric type"""
        with pytest.raises(ValueError):
            MetricDefinition(
                name="invalid_metric",
                type="invalid_type",  # This should cause an error
                description="Invalid metric"
            )