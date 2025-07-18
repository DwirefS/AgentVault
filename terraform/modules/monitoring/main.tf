# AgentVault™ Monitoring Module
# Comprehensive monitoring with Log Analytics, Application Insights, Prometheus, and Grafana

# Variables
variable "resource_group_name" {
  description = "Resource group name"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
}

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "log_analytics_retention_days" {
  description = "Log Analytics retention in days"
  type        = number
  default     = 90
}

variable "enable_application_insights" {
  description = "Enable Application Insights"
  type        = bool
  default     = true
}

variable "application_type" {
  description = "Application Insights type"
  type        = string
  default     = "web"
}

variable "enable_azure_monitor" {
  description = "Enable Azure Monitor"
  type        = bool
  default     = true
}

variable "create_action_groups" {
  description = "Create action groups"
  type        = bool
  default     = true
}

variable "action_groups" {
  description = "List of action groups"
  type = list(object({
    name              = string
    email_receivers   = list(string)
    sms_receivers     = list(string)
    webhook_receivers = list(string)
  }))
  default = []
}

variable "enable_prometheus" {
  description = "Enable Prometheus"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana"
  type        = bool
  default     = true
}

variable "grafana_admin_user" {
  description = "Grafana admin username"
  type        = string
  default     = "admin"
}

variable "create_dashboards" {
  description = "Create dashboards"
  type        = bool
  default     = true
}

variable "dashboard_templates" {
  description = "Dashboard templates to create"
  type        = list(string)
  default     = []
}

variable "create_workbooks" {
  description = "Create workbooks"
  type        = bool
  default     = true
}

variable "enable_container_insights" {
  description = "Enable container insights"
  type        = bool
  default     = true
}

variable "enable_vm_insights" {
  description = "Enable VM insights"
  type        = bool
  default     = true
}

variable "diagnostic_settings_enabled" {
  description = "Enable diagnostic settings"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.name_prefix}-law"
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = "PerGB2018"
  retention_in_days   = var.log_analytics_retention_days
  
  # Features
  daily_quota_gb                     = -1  # Unlimited
  internet_ingestion_enabled         = true
  internet_query_enabled             = true
  reservation_capacity_in_gb_per_day = 100
  
  tags = var.tags
}

# Log Analytics Solutions
resource "azurerm_log_analytics_solution" "container_insights" {
  count                 = var.enable_container_insights ? 1 : 0
  solution_name         = "ContainerInsights"
  location              = var.location
  resource_group_name   = var.resource_group_name
  workspace_resource_id = azurerm_log_analytics_workspace.main.id
  workspace_name        = azurerm_log_analytics_workspace.main.name
  
  plan {
    publisher = "Microsoft"
    product   = "OMSGallery/ContainerInsights"
  }
  
  tags = var.tags
}

resource "azurerm_log_analytics_solution" "vm_insights" {
  count                 = var.enable_vm_insights ? 1 : 0
  solution_name         = "VMInsights"
  location              = var.location
  resource_group_name   = var.resource_group_name
  workspace_resource_id = azurerm_log_analytics_workspace.main.id
  workspace_name        = azurerm_log_analytics_workspace.main.name
  
  plan {
    publisher = "Microsoft"
    product   = "OMSGallery/VMInsights"
  }
  
  tags = var.tags
}

resource "azurerm_log_analytics_solution" "security_center" {
  solution_name         = "SecurityCenterFree"
  location              = var.location
  resource_group_name   = var.resource_group_name
  workspace_resource_id = azurerm_log_analytics_workspace.main.id
  workspace_name        = azurerm_log_analytics_workspace.main.name
  
  plan {
    publisher = "Microsoft"
    product   = "OMSGallery/SecurityCenterFree"
  }
  
  tags = var.tags
}

# Application Insights
resource "azurerm_application_insights" "main" {
  count               = var.enable_application_insights ? 1 : 0
  name                = "${var.name_prefix}-ai"
  location            = var.location
  resource_group_name = var.resource_group_name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = var.application_type
  retention_in_days   = var.log_analytics_retention_days
  sampling_percentage = 100
  
  # Continuous export
  disable_ip_masking = false
  
  tags = var.tags
}

# Application Insights Smart Detection
resource "azurerm_application_insights_smart_detection_rule" "main" {
  for_each = var.enable_application_insights ? toset([
    "Slow page load time",
    "Slow server response time",
    "Long dependency duration",
    "Degradation in server response time",
    "Degradation in dependency duration",
    "Degradation in trace severity ratio",
    "Abnormal rise in exception volume",
    "Potential memory leak detected",
    "Potential security issue detected",
    "Abnormal rise in daily data volume"
  ]) : []
  
  name                    = each.value
  application_insights_id = azurerm_application_insights.main[0].id
  enabled                 = true
}

# Action Groups
resource "azurerm_monitor_action_group" "main" {
  for_each            = var.create_action_groups ? { for ag in var.action_groups : ag.name => ag } : {}
  name                = each.value.name
  resource_group_name = var.resource_group_name
  short_name          = substr(replace(each.value.name, "-", ""), 0, 12)
  
  dynamic "email_receiver" {
    for_each = each.value.email_receivers
    content {
      name                    = "email-${email_receiver.key}"
      email_address           = email_receiver.value
      use_common_alert_schema = true
    }
  }
  
  dynamic "sms_receiver" {
    for_each = each.value.sms_receivers
    content {
      name         = "sms-${sms_receiver.key}"
      country_code = "1"
      phone_number = sms_receiver.value
    }
  }
  
  dynamic "webhook_receiver" {
    for_each = each.value.webhook_receivers
    content {
      name                    = "webhook-${webhook_receiver.key}"
      service_uri             = webhook_receiver.value
      use_common_alert_schema = true
    }
  }
  
  tags = var.tags
}

# Azure Monitor Managed Prometheus
resource "azurerm_monitor_workspace" "prometheus" {
  count               = var.enable_prometheus ? 1 : 0
  name                = "${var.name_prefix}-prometheus"
  location            = var.location
  resource_group_name = var.resource_group_name
  
  tags = var.tags
}

# Azure Managed Grafana
resource "azurerm_dashboard_grafana" "main" {
  count                             = var.enable_grafana ? 1 : 0
  name                              = "${var.name_prefix}-grafana"
  location                          = var.location
  resource_group_name               = var.resource_group_name
  api_key_enabled                   = true
  deterministic_outbound_ip_enabled = true
  public_network_access_enabled     = true
  
  identity {
    type = "SystemAssigned"
  }
  
  azure_monitor_workspace_integrations {
    resource_id = var.enable_prometheus ? azurerm_monitor_workspace.prometheus[0].id : null
  }
  
  tags = var.tags
}

# Storage Account for diagnostics
resource "azurerm_storage_account" "diagnostics" {
  name                     = replace("${var.name_prefix}diag", "-", "")
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "GRS"
  min_tls_version         = "TLS1_2"
  
  blob_properties {
    delete_retention_policy {
      days = 30
    }
  }
  
  network_rules {
    default_action = "Allow"
    bypass         = ["AzureServices", "Logging", "Metrics"]
  }
  
  tags = var.tags
}

# Data Collection Rules
resource "azurerm_monitor_data_collection_rule" "main" {
  name                = "${var.name_prefix}-dcr"
  location            = var.location
  resource_group_name = var.resource_group_name
  
  destinations {
    log_analytics {
      workspace_resource_id = azurerm_log_analytics_workspace.main.id
      name                  = "log-analytics"
    }
    
    azure_monitor_metrics {
      name = "metrics"
    }
  }
  
  data_flow {
    streams      = ["Microsoft-ContainerLogV2", "Microsoft-KubeEvents", "Microsoft-KubePodInventory", "Microsoft-KubeNodeInventory"]
    destinations = ["log-analytics"]
  }
  
  data_flow {
    streams      = ["Microsoft-Perf", "Microsoft-InsightsMetrics"]
    destinations = ["metrics"]
  }
  
  data_sources {
    performance_counter {
      streams                       = ["Microsoft-Perf", "Microsoft-InsightsMetrics"]
      sampling_frequency_in_seconds = 60
      counter_specifiers = [
        "\\Processor Information(_Total)\\% Processor Time",
        "\\Processor Information(_Total)\\% Privileged Time",
        "\\Processor Information(_Total)\\% User Time",
        "\\Processor Information(_Total)\\Processor Frequency",
        "\\System\\Processes",
        "\\Process(_Total)\\Thread Count",
        "\\Process(_Total)\\Handle Count",
        "\\Memory\\% Committed Bytes In Use",
        "\\Memory\\Available Bytes",
        "\\Memory\\Committed Bytes",
        "\\Memory\\Cache Bytes",
        "\\PhysicalDisk(_Total)\\% Disk Time",
        "\\PhysicalDisk(_Total)\\% Disk Read Time",
        "\\PhysicalDisk(_Total)\\% Disk Write Time",
        "\\PhysicalDisk(_Total)\\Disk Transfers/sec",
        "\\PhysicalDisk(_Total)\\Disk Reads/sec",
        "\\PhysicalDisk(_Total)\\Disk Writes/sec",
        "\\Network Interface(*)\\Bytes Total/sec",
        "\\Network Interface(*)\\Bytes Sent/sec",
        "\\Network Interface(*)\\Bytes Received/sec",
        "\\Network Interface(*)\\Packets/sec",
        "\\Network Interface(*)\\Packets Sent/sec",
        "\\Network Interface(*)\\Packets Received/sec"
      ]
      name = "perfCounterDataSource"
    }
  }
  
  tags = var.tags
}

# Metric Alerts
resource "azurerm_monitor_metric_alert" "high_cpu" {
  name                = "${var.name_prefix}-high-cpu"
  resource_group_name = var.resource_group_name
  scopes              = [azurerm_log_analytics_workspace.main.id]
  description         = "Alert when CPU usage is high"
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT15M"
  
  criteria {
    metric_namespace = "Microsoft.OperationalInsights/workspaces"
    metric_name      = "Average_% Processor Time"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
  }
  
  dynamic "action" {
    for_each = var.create_action_groups ? var.action_groups : []
    content {
      action_group_id = azurerm_monitor_action_group.main[action.value.name].id
    }
  }
  
  tags = var.tags
}

resource "azurerm_monitor_metric_alert" "high_memory" {
  name                = "${var.name_prefix}-high-memory"
  resource_group_name = var.resource_group_name
  scopes              = [azurerm_log_analytics_workspace.main.id]
  description         = "Alert when memory usage is high"
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT15M"
  
  criteria {
    metric_namespace = "Microsoft.OperationalInsights/workspaces"
    metric_name      = "Average_% Committed Bytes In Use"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 90
  }
  
  dynamic "action" {
    for_each = var.create_action_groups ? var.action_groups : []
    content {
      action_group_id = azurerm_monitor_action_group.main[action.value.name].id
    }
  }
  
  tags = var.tags
}

# Workbooks
resource "azurerm_application_insights_workbook" "main" {
  for_each            = var.create_workbooks ? toset(var.dashboard_templates) : []
  name                = random_uuid.workbook[each.key].result
  location            = var.location
  resource_group_name = var.resource_group_name
  display_name        = "${var.name_prefix}-${each.value}"
  
  data_json = jsonencode({
    version = "Notebook/1.0"
    items = [{
      type = 1
      content = {
        json = "# ${each.value} Dashboard\n\nThis workbook provides insights into ${each.value}."
      }
    }]
    isLocked = false
  })
  
  tags = var.tags
}

resource "random_uuid" "workbook" {
  for_each = var.create_workbooks ? toset(var.dashboard_templates) : []
}

# Outputs
output "log_analytics_workspace_id" {
  value = azurerm_log_analytics_workspace.main.id
}

output "log_analytics_workspace_name" {
  value = azurerm_log_analytics_workspace.main.name
}

output "log_analytics_workspace_key" {
  value     = azurerm_log_analytics_workspace.main.primary_shared_key
  sensitive = true
}

output "application_insights_id" {
  value = var.enable_application_insights ? azurerm_application_insights.main[0].id : ""
}

output "application_insights_instrumentation_key" {
  value     = var.enable_application_insights ? azurerm_application_insights.main[0].instrumentation_key : ""
  sensitive = true
}

output "application_insights_connection_string" {
  value     = var.enable_application_insights ? azurerm_application_insights.main[0].connection_string : ""
  sensitive = true
}

output "prometheus_endpoint" {
  value = var.enable_prometheus ? "https://${azurerm_monitor_workspace.prometheus[0].default_data_collection_endpoint_id}" : ""
}

output "grafana_endpoint" {
  value = var.enable_grafana ? azurerm_dashboard_grafana.main[0].endpoint : ""
}

output "storage_account_id" {
  value = azurerm_storage_account.diagnostics.id
}

output "action_group_ids" {
  value = { for k, v in azurerm_monitor_action_group.main : k => v.id }
}

output "data_collection_rule_id" {
  value = azurerm_monitor_data_collection_rule.main.id
}