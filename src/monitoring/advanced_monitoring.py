"""
AgentVault™ Advanced Monitoring and Alerting System
Production-ready observability with custom metrics and intelligent alerting
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
import aiohttp
from azure.monitor.query import LogsQueryClient, MetricsQueryClient
from azure.monitor.ingestion import LogsIngestionClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert states"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class MetricDefinition:
    """Definition of a custom metric"""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    unit: str = ""
    buckets: Optional[List[float]] = None
    quantiles: Optional[List[float]] = None


@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    expression: str
    duration: timedelta
    severity: AlertSeverity
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    cooldown: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    actions: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """Active alert instance"""
    alert_id: str
    rule: AlertRule
    state: AlertState
    fired_at: datetime
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    message: str = ""
    fingerprint: str = ""


@dataclass
class MetricSample:
    """Single metric sample"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    metric_name: str
    timestamp: datetime
    value: float
    expected_value: float
    deviation: float
    z_score: float
    is_anomaly: bool
    confidence: float


class AdvancedMonitoringSystem:
    """
    Advanced monitoring system with:
    - Custom metrics collection and aggregation
    - Intelligent alerting with ML-based thresholds
    - Anomaly detection
    - Predictive alerting
    - Azure Monitor integration
    - Multi-channel notifications
    - Metric correlation analysis
    - SLA tracking and reporting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Azure Monitor configuration
        self.workspace_id = config.get('azure_monitor_workspace_id')
        self.credential = DefaultAzureCredential()
        
        if self.workspace_id:
            self.logs_client = LogsQueryClient(self.credential)
            self.metrics_client = MetricsQueryClient(self.credential)
            self.ingestion_client = LogsIngestionClient(
                endpoint=config.get('data_collection_endpoint'),
                credential=self.credential
            )
        
        # Metric definitions
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.metrics: Dict[str, Any] = {}  # Prometheus metrics
        
        # Time series data for analysis
        self.time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Alert rules and active alerts
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Anomaly detection
        self.anomaly_detectors: Dict[str, Any] = {}
        self.anomaly_threshold = config.get('anomaly_threshold', 3.0)  # Z-score threshold
        
        # Notification channels
        self.notification_channels: Dict[str, Callable] = {}
        self._setup_notification_channels()
        
        # SLA tracking
        self.sla_targets: Dict[str, float] = config.get('sla_targets', {})
        self.sla_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Correlation analysis
        self.correlation_window = timedelta(minutes=30)
        self.correlation_threshold = 0.7
        
        # Background tasks
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
        # Initialize default metrics
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default system metrics"""
        
        # Request metrics
        self.register_metric(MetricDefinition(
            name="agentvault_requests_total",
            type=MetricType.COUNTER,
            description="Total number of requests",
            labels=["agent_id", "operation", "status"]
        ))
        
        self.register_metric(MetricDefinition(
            name="agentvault_request_duration_seconds",
            type=MetricType.HISTOGRAM,
            description="Request duration in seconds",
            labels=["operation", "tier"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        ))
        
        # Storage metrics
        self.register_metric(MetricDefinition(
            name="agentvault_storage_usage_bytes",
            type=MetricType.GAUGE,
            description="Storage usage in bytes",
            labels=["agent_id", "tier"],
            unit="bytes"
        ))
        
        self.register_metric(MetricDefinition(
            name="agentvault_storage_operations_total",
            type=MetricType.COUNTER,
            description="Total storage operations",
            labels=["operation", "tier", "status"]
        ))
        
        # Cache metrics
        self.register_metric(MetricDefinition(
            name="agentvault_cache_hit_rate",
            type=MetricType.GAUGE,
            description="Cache hit rate",
            labels=["cache_type"]
        ))
        
        # ML metrics
        self.register_metric(MetricDefinition(
            name="agentvault_ml_prediction_accuracy",
            type=MetricType.GAUGE,
            description="ML model prediction accuracy",
            labels=["model_type"]
        ))
        
        self.register_metric(MetricDefinition(
            name="agentvault_ml_inference_duration_seconds",
            type=MetricType.HISTOGRAM,
            description="ML inference duration",
            labels=["model_type"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
        ))
        
        # System metrics
        self.register_metric(MetricDefinition(
            name="agentvault_active_agents",
            type=MetricType.GAUGE,
            description="Number of active agents",
            labels=["framework"]
        ))
        
        self.register_metric(MetricDefinition(
            name="agentvault_error_rate",
            type=MetricType.GAUGE,
            description="Error rate",
            labels=["error_type"]
        ))
        
        # SLA metrics
        self.register_metric(MetricDefinition(
            name="agentvault_sla_compliance",
            type=MetricType.GAUGE,
            description="SLA compliance percentage",
            labels=["sla_type"]
        ))
    
    def register_metric(self, definition: MetricDefinition) -> None:
        """Register a custom metric"""
        
        self.metric_definitions[definition.name] = definition
        
        # Create Prometheus metric
        if definition.type == MetricType.COUNTER:
            self.metrics[definition.name] = Counter(
                definition.name,
                definition.description,
                labelnames=definition.labels
            )
        elif definition.type == MetricType.GAUGE:
            self.metrics[definition.name] = Gauge(
                definition.name,
                definition.description,
                labelnames=definition.labels
            )
        elif definition.type == MetricType.HISTOGRAM:
            self.metrics[definition.name] = Histogram(
                definition.name,
                definition.description,
                labelnames=definition.labels,
                buckets=definition.buckets or Histogram.DEFAULT_BUCKETS
            )
        elif definition.type == MetricType.SUMMARY:
            self.metrics[definition.name] = Summary(
                definition.name,
                definition.description,
                labelnames=definition.labels
            )
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric value"""
        
        if metric_name not in self.metrics:
            logger.warning(f"Metric {metric_name} not registered")
            return
        
        metric = self.metrics[metric_name]
        definition = self.metric_definitions[metric_name]
        labels = labels or {}
        
        # Record to Prometheus
        if definition.type == MetricType.COUNTER:
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
        elif definition.type == MetricType.GAUGE:
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
        elif definition.type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
        
        # Store time series data
        sample = MetricSample(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels
        )
        
        series_key = self._get_series_key(metric_name, labels)
        self.time_series[series_key].append(sample)
        
        # Check alert rules
        asyncio.create_task(self._evaluate_alerts(metric_name, value, labels))
        
        # Detect anomalies
        asyncio.create_task(self._detect_anomalies(metric_name, value, labels))
    
    def _get_series_key(self, metric_name: str, labels: Dict[str, str]) -> str:
        """Generate time series key from metric name and labels"""
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{metric_name}{{{label_str}}}"
    
    async def add_alert_rule(self, rule: AlertRule) -> None:
        """Add or update alert rule"""
        
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
        
        # Evaluate immediately
        await self._evaluate_rule(rule)
    
    async def _evaluate_alerts(
        self,
        metric_name: str,
        value: float,
        labels: Dict[str, str]
    ) -> None:
        """Evaluate alert rules for a metric"""
        
        for rule in self.alert_rules.values():
            # Simple expression matching (in production, use proper expression parser)
            if metric_name in rule.expression:
                await self._evaluate_rule(rule, metric_name, value, labels)
    
    async def _evaluate_rule(
        self,
        rule: AlertRule,
        metric_name: Optional[str] = None,
        value: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Evaluate a single alert rule"""
        
        try:
            # Parse and evaluate expression
            # This is simplified - in production, use proper expression evaluator
            result = await self._evaluate_expression(rule.expression, metric_name, value, labels)
            
            if result['firing']:
                await self._handle_alert_firing(rule, result['value'], labels or {})
            else:
                await self._handle_alert_resolved(rule)
                
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.rule_id}: {str(e)}")
    
    async def _evaluate_expression(
        self,
        expression: str,
        metric_name: Optional[str] = None,
        value: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Evaluate alert expression"""
        
        # Simplified expression evaluation
        # In production, implement proper PromQL-like expression parser
        
        # Example: "agentvault_error_rate > 0.05"
        parts = expression.split()
        if len(parts) == 3:
            target_metric, operator, threshold = parts
            threshold = float(threshold)
            
            if metric_name == target_metric:
                current_value = value
            else:
                # Get latest value from time series
                series_key = self._get_series_key(target_metric, labels or {})
                series = self.time_series.get(series_key, [])
                if series:
                    current_value = series[-1].value
                else:
                    return {'firing': False, 'value': 0}
            
            # Evaluate condition
            if operator == '>':
                firing = current_value > threshold
            elif operator == '<':
                firing = current_value < threshold
            elif operator == '>=':
                firing = current_value >= threshold
            elif operator == '<=':
                firing = current_value <= threshold
            elif operator == '==':
                firing = current_value == threshold
            else:
                firing = False
            
            return {'firing': firing, 'value': current_value}
        
        return {'firing': False, 'value': 0}
    
    async def _handle_alert_firing(
        self,
        rule: AlertRule,
        value: float,
        labels: Dict[str, str]
    ) -> None:
        """Handle alert firing"""
        
        # Generate fingerprint for deduplication
        fingerprint = self._generate_alert_fingerprint(rule, labels)
        
        # Check if alert already active
        if fingerprint in self.active_alerts:
            alert = self.active_alerts[fingerprint]
            # Update value but don't re-notify if in cooldown
            alert.value = value
            
            if datetime.utcnow() - alert.fired_at < rule.cooldown:
                return
        else:
            # Create new alert
            alert = Alert(
                alert_id=f"{rule.rule_id}-{datetime.utcnow().timestamp()}",
                rule=rule,
                state=AlertState.FIRING,
                fired_at=datetime.utcnow(),
                labels={**rule.labels, **labels},
                value=value,
                message=self._format_alert_message(rule, value, labels),
                fingerprint=fingerprint
            )
            
            self.active_alerts[fingerprint] = alert
            self.alert_history.append(alert)
        
        # Send notifications
        await self._send_notifications(alert)
        
        logger.warning(f"Alert firing: {rule.name} (value={value})")
    
    async def _handle_alert_resolved(self, rule: AlertRule) -> None:
        """Handle alert resolution"""
        
        # Find active alerts for this rule
        resolved = []
        for fingerprint, alert in self.active_alerts.items():
            if alert.rule.rule_id == rule.rule_id and alert.state == AlertState.FIRING:
                alert.state = AlertState.RESOLVED
                alert.resolved_at = datetime.utcnow()
                resolved.append(fingerprint)
                
                # Send resolution notification
                await self._send_notifications(alert, resolved=True)
                
                logger.info(f"Alert resolved: {rule.name}")
        
        # Remove resolved alerts
        for fingerprint in resolved:
            del self.active_alerts[fingerprint]
    
    def _generate_alert_fingerprint(
        self,
        rule: AlertRule,
        labels: Dict[str, str]
    ) -> str:
        """Generate unique fingerprint for alert deduplication"""
        
        fingerprint_data = {
            'rule_id': rule.rule_id,
            'labels': labels
        }
        
        import hashlib
        return hashlib.sha256(
            json.dumps(fingerprint_data, sort_keys=True).encode()
        ).hexdigest()
    
    def _format_alert_message(
        self,
        rule: AlertRule,
        value: float,
        labels: Dict[str, str]
    ) -> str:
        """Format alert message"""
        
        message = rule.annotations.get('summary', f"Alert: {rule.name}")
        message += f"\n\nCurrent value: {value}"
        
        if rule.annotations.get('description'):
            message += f"\n\n{rule.annotations['description']}"
        
        if labels:
            message += "\n\nLabels:"
            for k, v in labels.items():
                message += f"\n  {k}: {v}"
        
        return message
    
    async def _send_notifications(
        self,
        alert: Alert,
        resolved: bool = False
    ) -> None:
        """Send alert notifications"""
        
        for action in alert.rule.actions:
            if action in self.notification_channels:
                try:
                    await self.notification_channels[action](alert, resolved)
                except Exception as e:
                    logger.error(f"Failed to send notification via {action}: {str(e)}")
    
    def _setup_notification_channels(self) -> None:
        """Setup notification channels"""
        
        # Webhook notifications
        if self.config.get('webhook_url'):
            self.notification_channels['webhook'] = self._send_webhook_notification
        
        # Email notifications
        if self.config.get('smtp_config'):
            self.notification_channels['email'] = self._send_email_notification
        
        # Slack notifications
        if self.config.get('slack_webhook'):
            self.notification_channels['slack'] = self._send_slack_notification
        
        # PagerDuty integration
        if self.config.get('pagerduty_key'):
            self.notification_channels['pagerduty'] = self._send_pagerduty_notification
        
        # Azure Monitor alerts
        if self.config.get('azure_action_group'):
            self.notification_channels['azure'] = self._send_azure_alert
    
    async def _send_webhook_notification(
        self,
        alert: Alert,
        resolved: bool = False
    ) -> None:
        """Send webhook notification"""
        
        webhook_url = self.config.get('webhook_url')
        
        payload = {
            'alert_id': alert.alert_id,
            'name': alert.rule.name,
            'severity': alert.rule.severity.value,
            'state': 'resolved' if resolved else 'firing',
            'value': alert.value,
            'message': alert.message,
            'labels': alert.labels,
            'fired_at': alert.fired_at.isoformat(),
            'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Webhook notification failed: {response.status}")
    
    async def _send_slack_notification(
        self,
        alert: Alert,
        resolved: bool = False
    ) -> None:
        """Send Slack notification"""
        
        slack_webhook = self.config.get('slack_webhook')
        
        # Format message for Slack
        color = {
            AlertSeverity.INFO: '#36a64f',
            AlertSeverity.WARNING: '#ff9900',
            AlertSeverity.ERROR: '#ff0000',
            AlertSeverity.CRITICAL: '#990000'
        }.get(alert.rule.severity, '#808080')
        
        attachment = {
            'color': color,
            'title': f"{'✅ RESOLVED' if resolved else '🚨 FIRING'}: {alert.rule.name}",
            'text': alert.message,
            'fields': [
                {'title': 'Severity', 'value': alert.rule.severity.value, 'short': True},
                {'title': 'Value', 'value': str(alert.value), 'short': True}
            ],
            'ts': int(alert.fired_at.timestamp())
        }
        
        if alert.labels:
            attachment['fields'].extend([
                {'title': k, 'value': v, 'short': True}
                for k, v in alert.labels.items()
            ])
        
        payload = {'attachments': [attachment]}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(slack_webhook, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Slack notification failed: {response.status}")
    
    async def _send_email_notification(
        self,
        alert: Alert,
        resolved: bool = False
    ) -> None:
        """Send email notification"""
        
        # Email implementation would go here
        # Using SMTP or cloud email service
        pass
    
    async def _send_pagerduty_notification(
        self,
        alert: Alert,
        resolved: bool = False
    ) -> None:
        """Send PagerDuty notification"""
        
        # PagerDuty Events API v2 implementation
        pass
    
    async def _send_azure_alert(
        self,
        alert: Alert,
        resolved: bool = False
    ) -> None:
        """Send Azure Monitor alert"""
        
        # Azure Monitor alert implementation
        pass
    
    async def _detect_anomalies(
        self,
        metric_name: str,
        value: float,
        labels: Dict[str, str]
    ) -> None:
        """Detect anomalies in metric values"""
        
        series_key = self._get_series_key(metric_name, labels)
        series = list(self.time_series.get(series_key, []))
        
        if len(series) < 30:  # Need minimum samples
            return
        
        # Get recent values
        recent_values = [s.value for s in series[-100:]]
        
        # Calculate statistics
        mean = np.mean(recent_values)
        std = np.std(recent_values)
        
        if std == 0:  # No variation
            return
        
        # Calculate z-score
        z_score = (value - mean) / std
        
        # Detect anomaly
        is_anomaly = abs(z_score) > self.anomaly_threshold
        
        if is_anomaly:
            anomaly = AnomalyDetection(
                metric_name=metric_name,
                timestamp=datetime.utcnow(),
                value=value,
                expected_value=mean,
                deviation=abs(value - mean),
                z_score=z_score,
                is_anomaly=True,
                confidence=min(abs(z_score) / self.anomaly_threshold, 1.0)
            )
            
            # Create anomaly alert
            await self._create_anomaly_alert(anomaly, labels)
            
            logger.warning(
                f"Anomaly detected in {metric_name}: "
                f"value={value}, expected={mean:.2f}, z_score={z_score:.2f}"
            )
    
    async def _create_anomaly_alert(
        self,
        anomaly: AnomalyDetection,
        labels: Dict[str, str]
    ) -> None:
        """Create alert for detected anomaly"""
        
        # Create dynamic alert rule
        rule = AlertRule(
            rule_id=f"anomaly-{anomaly.metric_name}-{datetime.utcnow().timestamp()}",
            name=f"Anomaly in {anomaly.metric_name}",
            expression=f"{anomaly.metric_name} deviates significantly",
            duration=timedelta(minutes=0),
            severity=AlertSeverity.WARNING if abs(anomaly.z_score) < 4 else AlertSeverity.ERROR,
            labels={'type': 'anomaly', 'metric': anomaly.metric_name},
            annotations={
                'summary': f"Anomaly detected in {anomaly.metric_name}",
                'description': (
                    f"Current value {anomaly.value} deviates from expected "
                    f"{anomaly.expected_value:.2f} by {anomaly.z_score:.1f} standard deviations"
                )
            },
            actions=['webhook', 'slack']
        )
        
        # Create alert
        await self._handle_alert_firing(rule, anomaly.value, labels)
    
    async def get_metric_forecast(
        self,
        metric_name: str,
        labels: Dict[str, str],
        horizon_minutes: int = 60
    ) -> List[Tuple[datetime, float, float]]:
        """Forecast metric values"""
        
        series_key = self._get_series_key(metric_name, labels)
        series = list(self.time_series.get(series_key, []))
        
        if len(series) < 100:
            return []
        
        # Simple time series forecasting
        # In production, use proper forecasting models (Prophet, ARIMA, etc.)
        
        values = [s.value for s in series]
        timestamps = [s.timestamp for s in series]
        
        # Calculate trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Calculate seasonality (simplified)
        detrended = values - (slope * x + intercept)
        seasonal_period = 60  # Assume hourly seasonality
        
        # Generate forecast
        forecast = []
        current_time = datetime.utcnow()
        
        for i in range(horizon_minutes):
            future_time = current_time + timedelta(minutes=i)
            
            # Trend component
            trend_value = slope * (len(values) + i) + intercept
            
            # Add seasonality
            seasonal_index = i % seasonal_period
            if seasonal_index < len(detrended):
                seasonal_value = detrended[seasonal_index]
            else:
                seasonal_value = 0
            
            # Combine components
            forecast_value = trend_value + seasonal_value
            
            # Calculate confidence interval (simplified)
            std_error = np.std(detrended) * np.sqrt(1 + i / len(values))
            lower_bound = forecast_value - 2 * std_error
            upper_bound = forecast_value + 2 * std_error
            
            forecast.append((future_time, forecast_value, upper_bound))
        
        return forecast
    
    async def analyze_metric_correlation(
        self,
        metric1: str,
        metric2: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Analyze correlation between two metrics"""
        
        series1_key = self._get_series_key(metric1, labels or {})
        series2_key = self._get_series_key(metric2, labels or {})
        
        series1 = list(self.time_series.get(series1_key, []))
        series2 = list(self.time_series.get(series2_key, []))
        
        if len(series1) < 30 or len(series2) < 30:
            return {'correlation': 0, 'significant': False}
        
        # Align time series by timestamp
        df1 = pd.DataFrame([(s.timestamp, s.value) for s in series1], columns=['time', 'value1'])
        df2 = pd.DataFrame([(s.timestamp, s.value) for s in series2], columns=['time', 'value2'])
        
        # Merge on nearest timestamp
        df1.set_index('time', inplace=True)
        df2.set_index('time', inplace=True)
        
        merged = pd.merge_asof(
            df1.sort_index(),
            df2.sort_index(),
            left_index=True,
            right_index=True,
            direction='nearest',
            tolerance=pd.Timedelta('1min')
        )
        
        if len(merged) < 30:
            return {'correlation': 0, 'significant': False}
        
        # Calculate correlation
        correlation = merged['value1'].corr(merged['value2'])
        
        # Test significance
        n = len(merged)
        t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        return {
            'correlation': correlation,
            'significant': p_value < 0.05,
            'p_value': p_value,
            'sample_size': n,
            'interpretation': self._interpret_correlation(correlation)
        }
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient"""
        
        abs_corr = abs(correlation)
        
        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.5:
            strength = "moderate"
        elif abs_corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if correlation > 0 else "negative"
        
        return f"{strength} {direction} correlation"
    
    async def get_sla_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Generate SLA compliance report"""
        
        report = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'sla_metrics': {},
            'overall_compliance': 0.0
        }
        
        compliance_scores = []
        
        for sla_name, target_value in self.sla_targets.items():
            # Get metric data for period
            series_key = f"agentvault_sla_{sla_name}"
            series = [
                s for s in self.time_series.get(series_key, [])
                if start_time <= s.timestamp <= end_time
            ]
            
            if not series:
                continue
            
            # Calculate compliance
            values = [s.value for s in series]
            
            if 'uptime' in sla_name:
                # For uptime, calculate percentage above threshold
                compliant_samples = sum(1 for v in values if v >= target_value)
                compliance = compliant_samples / len(values) * 100
            elif 'latency' in sla_name:
                # For latency, calculate percentage below threshold
                compliant_samples = sum(1 for v in values if v <= target_value)
                compliance = compliant_samples / len(values) * 100
            else:
                # Generic comparison
                compliant_samples = sum(1 for v in values if v >= target_value)
                compliance = compliant_samples / len(values) * 100
            
            report['sla_metrics'][sla_name] = {
                'target': target_value,
                'achieved': np.mean(values),
                'compliance_percentage': compliance,
                'violations': len(values) - compliant_samples,
                'total_samples': len(values)
            }
            
            compliance_scores.append(compliance)
        
        # Calculate overall compliance
        if compliance_scores:
            report['overall_compliance'] = np.mean(compliance_scores)
        
        return report
    
    async def export_metrics(
        self,
        metric_names: List[str],
        start_time: datetime,
        end_time: datetime,
        format: str = 'json'
    ) -> Any:
        """Export metrics data"""
        
        data = {}
        
        for metric_name in metric_names:
            metric_data = []
            
            # Get all series for this metric
            for series_key, series in self.time_series.items():
                if series_key.startswith(metric_name):
                    filtered_series = [
                        s for s in series
                        if start_time <= s.timestamp <= end_time
                    ]
                    
                    if filtered_series:
                        metric_data.append({
                            'series': series_key,
                            'samples': [
                                {
                                    'timestamp': s.timestamp.isoformat(),
                                    'value': s.value,
                                    'labels': s.labels
                                }
                                for s in filtered_series
                            ]
                        })
            
            if metric_data:
                data[metric_name] = metric_data
        
        if format == 'json':
            return json.dumps(data, indent=2)
        elif format == 'csv':
            # Convert to CSV format
            rows = []
            for metric_name, series_list in data.items():
                for series_data in series_list:
                    for sample in series_data['samples']:
                        row = {
                            'metric': metric_name,
                            'series': series_data['series'],
                            'timestamp': sample['timestamp'],
                            'value': sample['value']
                        }
                        row.update(sample['labels'])
                        rows.append(row)
            
            df = pd.DataFrame(rows)
            return df.to_csv(index=False)
        else:
            return data
    
    async def query_azure_monitor(
        self,
        query: str,
        timespan: timedelta
    ) -> List[Dict[str, Any]]:
        """Query Azure Monitor logs"""
        
        if not self.logs_client:
            return []
        
        try:
            response = self.logs_client.query_workspace(
                workspace_id=self.workspace_id,
                query=query,
                timespan=timespan
            )
            
            # Convert to list of dicts
            results = []
            for table in response.tables:
                for row in table.rows:
                    results.append(dict(zip(table.columns, row)))
            
            return results
            
        except AzureError as e:
            logger.error(f"Azure Monitor query failed: {str(e)}")
            return []
    
    async def ingest_custom_logs(
        self,
        logs: List[Dict[str, Any]],
        stream_name: str
    ) -> bool:
        """Ingest custom logs to Azure Monitor"""
        
        if not self.ingestion_client:
            return False
        
        try:
            # Prepare logs for ingestion
            prepared_logs = []
            for log in logs:
                prepared_log = {
                    'TimeGenerated': log.get('timestamp', datetime.utcnow()).isoformat(),
                    'Level': log.get('level', 'INFO'),
                    'Message': log.get('message', ''),
                    'Properties': json.dumps(log.get('properties', {}))
                }
                prepared_logs.append(prepared_log)
            
            # Ingest logs
            self.ingestion_client.upload(
                rule_id=self.config.get('data_collection_rule_id'),
                stream_name=stream_name,
                logs=prepared_logs
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Log ingestion failed: {str(e)}")
            return False
    
    def _start_background_tasks(self) -> None:
        """Start background monitoring tasks"""
        
        self._running = True
        self._background_tasks = [
            asyncio.create_task(self._alert_evaluator()),
            asyncio.create_task(self._metric_aggregator()),
            asyncio.create_task(self._anomaly_detector()),
            asyncio.create_task(self._sla_tracker()),
            asyncio.create_task(self._metric_exporter())
        ]
    
    async def _alert_evaluator(self) -> None:
        """Continuously evaluate alert rules"""
        
        while self._running:
            try:
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
                for rule in self.alert_rules.values():
                    await self._evaluate_rule(rule)
                
            except Exception as e:
                logger.error(f"Alert evaluator error: {str(e)}")
    
    async def _metric_aggregator(self) -> None:
        """Aggregate metrics for efficiency"""
        
        while self._running:
            try:
                await asyncio.sleep(60)  # Aggregate every minute
                
                # Calculate derived metrics
                await self._calculate_derived_metrics()
                
                # Clean old time series data
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                for series_key, series in self.time_series.items():
                    while series and series[0].timestamp < cutoff_time:
                        series.popleft()
                
            except Exception as e:
                logger.error(f"Metric aggregator error: {str(e)}")
    
    async def _calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from raw metrics"""
        
        # Calculate error rate
        total_requests = sum(
            s[-1].value for k, s in self.time_series.items()
            if k.startswith('agentvault_requests_total') and s
        )
        
        error_requests = sum(
            s[-1].value for k, s in self.time_series.items()
            if 'status=error' in k and s
        )
        
        if total_requests > 0:
            error_rate = error_requests / total_requests
            self.record_metric('agentvault_error_rate', error_rate, {'error_type': 'all'})
        
        # Calculate SLA compliance
        for sla_name, target_value in self.sla_targets.items():
            series_key = f"agentvault_{sla_name}"
            series = list(self.time_series.get(series_key, []))[-100:]  # Last 100 samples
            
            if series:
                values = [s.value for s in series]
                
                if 'latency' in sla_name:
                    # For latency, count samples below target
                    compliance = sum(1 for v in values if v <= target_value) / len(values)
                else:
                    # For others, count samples above target
                    compliance = sum(1 for v in values if v >= target_value) / len(values)
                
                self.record_metric(
                    'agentvault_sla_compliance',
                    compliance * 100,
                    {'sla_type': sla_name}
                )
    
    async def _anomaly_detector(self) -> None:
        """Background anomaly detection"""
        
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Run anomaly detection on all metrics
                for series_key, series in self.time_series.items():
                    if len(series) >= 100:
                        # Use more sophisticated anomaly detection
                        await self._detect_advanced_anomalies(series_key, series)
                
            except Exception as e:
                logger.error(f"Anomaly detector error: {str(e)}")
    
    async def _detect_advanced_anomalies(
        self,
        series_key: str,
        series: deque
    ) -> None:
        """Advanced anomaly detection using multiple methods"""
        
        values = [s.value for s in series]
        
        # Method 1: Isolation Forest
        # Method 2: LSTM Autoencoder
        # Method 3: Statistical Process Control
        
        # For now, use enhanced statistical method
        window_size = 30
        if len(values) < window_size:
            return
        
        # Calculate rolling statistics
        rolling_mean = pd.Series(values).rolling(window=window_size).mean()
        rolling_std = pd.Series(values).rolling(window=window_size).std()
        
        # Check last value
        if pd.notna(rolling_mean.iloc[-1]) and pd.notna(rolling_std.iloc[-1]):
            current_value = values[-1]
            expected = rolling_mean.iloc[-1]
            std = rolling_std.iloc[-1]
            
            if std > 0:
                z_score = (current_value - expected) / std
                
                # Dynamic threshold based on metric stability
                cv = std / expected if expected != 0 else 0  # Coefficient of variation
                threshold = self.anomaly_threshold * (1 + cv)  # Adjust threshold
                
                if abs(z_score) > threshold:
                    # Extract metric name and labels from series key
                    parts = series_key.split('{')
                    metric_name = parts[0]
                    
                    anomaly = AnomalyDetection(
                        metric_name=metric_name,
                        timestamp=series[-1].timestamp,
                        value=current_value,
                        expected_value=expected,
                        deviation=abs(current_value - expected),
                        z_score=z_score,
                        is_anomaly=True,
                        confidence=min(abs(z_score) / threshold, 1.0)
                    )
                    
                    # Log for analysis
                    logger.info(f"Advanced anomaly detected: {anomaly}")
    
    async def _sla_tracker(self) -> None:
        """Track SLA compliance"""
        
        while self._running:
            try:
                await asyncio.sleep(3600)  # Check hourly
                
                # Generate SLA report for last hour
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=1)
                
                report = await self.get_sla_report(start_time, end_time)
                
                # Check if any SLA is violated
                for sla_name, metrics in report['sla_metrics'].items():
                    if metrics['compliance_percentage'] < 99.9:  # SLA threshold
                        # Create SLA violation alert
                        rule = AlertRule(
                            rule_id=f"sla-violation-{sla_name}",
                            name=f"SLA Violation: {sla_name}",
                            expression=f"sla_compliance < 99.9",
                            duration=timedelta(minutes=0),
                            severity=AlertSeverity.ERROR,
                            labels={'sla': sla_name},
                            annotations={
                                'summary': f"SLA {sla_name} violated",
                                'description': f"Compliance: {metrics['compliance_percentage']:.1f}%"
                            },
                            actions=['webhook', 'slack', 'pagerduty']
                        )
                        
                        await self._handle_alert_firing(
                            rule,
                            metrics['compliance_percentage'],
                            {'sla': sla_name}
                        )
                
            except Exception as e:
                logger.error(f"SLA tracker error: {str(e)}")
    
    async def _metric_exporter(self) -> None:
        """Export metrics to external systems"""
        
        while self._running:
            try:
                await asyncio.sleep(300)  # Export every 5 minutes
                
                # Export to Azure Monitor
                if self.config.get('azure_monitor_export'):
                    await self._export_to_azure_monitor()
                
                # Export to other systems (Datadog, New Relic, etc.)
                
            except Exception as e:
                logger.error(f"Metric exporter error: {str(e)}")
    
    async def _export_to_azure_monitor(self) -> None:
        """Export metrics to Azure Monitor"""
        
        # Prepare custom metrics
        custom_metrics = []
        
        for series_key, series in self.time_series.items():
            if series:
                latest = series[-1]
                
                # Parse series key
                parts = series_key.split('{')
                metric_name = parts[0]
                
                custom_metrics.append({
                    'metric': metric_name,
                    'value': latest.value,
                    'timestamp': latest.timestamp.isoformat(),
                    'dimensions': latest.labels
                })
        
        # Send to Azure Monitor
        # Implementation depends on Azure Monitor API
        pass
    
    async def shutdown(self) -> None:
        """Shutdown monitoring system"""
        
        logger.info("Shutting down Advanced Monitoring System...")
        
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Export final metrics
        await self._export_to_azure_monitor()
        
        logger.info("Advanced Monitoring System shutdown complete")