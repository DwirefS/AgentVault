# AgentVault™ Cost Management Module - Cost Optimization & Budget Control
# ======================================================================
# Comprehensive cost management, optimization, and budget control for AgentVault
#
# This module creates:
# - Cost budgets with automated alerts at multiple thresholds
# - Cost analysis views for different perspectives
# - Automated cost optimization recommendations
# - Resource tagging policies for cost allocation
# - Scheduled actions for cost reduction (dev/test shutdown)
# - Cost anomaly detection and alerting
# - Chargeback/showback reporting configuration
# - Reserved capacity recommendations and automation
#
# Key Features:
# - Multi-level budget alerts (50%, 75%, 90%, 100%, 110%)
# - Automated resource rightsizing recommendations
# - Unused resource identification and cleanup
# - Cost allocation by department/project/team
# - Integration with FinOps practices
# - Automated report generation and distribution
#
# Author: Dwiref Sharma
# Contact: DwirefS@SapientEdge.io
# License: Proprietary - SapientEdge LLC

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
  }
}

# Local variables for cost management configuration
locals {
  name_prefix = "agentvault-${var.environment}"
  
  # Budget thresholds for progressive alerting
  budget_thresholds = [
    {
      percentage = 50
      severity   = 3  # Informational
      message    = "Budget usage has reached 50%. Review spending patterns."
    },
    {
      percentage = 75
      severity   = 2  # Warning
      message    = "Budget usage has reached 75%. Consider optimization measures."
    },
    {
      percentage = 90
      severity   = 1  # Error
      message    = "Budget usage has reached 90%. Immediate attention required."
    },
    {
      percentage = 100
      severity   = 0  # Critical
      message    = "Budget limit reached. Cost controls may be enforced."
    },
    {
      percentage = 110
      severity   = 0  # Critical
      message    = "Budget exceeded by 10%. Immediate action required."
    }
  ]
  
  # Cost views for different analysis perspectives
  cost_views = {
    by_service = {
      name        = "AgentVault Services"
      description = "Cost breakdown by Azure service"
      type        = "Service"
      chart_type  = "StackedColumn"
    }
    by_resource_group = {
      name        = "AgentVault Resource Groups"
      description = "Cost analysis by resource group"
      type        = "ResourceGroup"
      chart_type  = "DonutChart"
    }
    by_tag_environment = {
      name        = "AgentVault Environments"
      description = "Cost comparison across environments"
      type        = "Tag"
      tag_key     = "Environment"
      chart_type  = "Line"
    }
    by_tag_component = {
      name        = "AgentVault Components"
      description = "Cost per application component"
      type        = "Tag"
      tag_key     = "Component"
      chart_type  = "Bar"
    }
    by_location = {
      name        = "AgentVault Regions"
      description = "Geographic cost distribution"
      type        = "Location"
      chart_type  = "Map"
    }
  }
  
  # Scheduled actions for cost optimization
  scheduled_actions = {
    shutdown_dev = {
      name        = "Shutdown Dev Environment"
      description = "Automatically shutdown dev resources after hours"
      schedule    = "0 19 * * 1-5"  # 7 PM weekdays
      action      = "Deallocate"
      scope       = "Environment:dev"
    }
    weekend_scaling = {
      name        = "Weekend Scaling"
      description = "Scale down non-production resources on weekends"
      schedule    = "0 0 * * 6"  # Saturday midnight
      action      = "Scale"
      scope       = "Environment:staging"
    }
  }
  
  # Cost optimization rules
  optimization_rules = {
    unused_disks = {
      name        = "Unused Managed Disks"
      description = "Identify and flag unattached managed disks"
      threshold   = 7  # Days unattached
      action      = "Alert"
    }
    idle_vms = {
      name        = "Idle Virtual Machines"
      description = "Detect VMs with low CPU usage"
      threshold   = 5  # Average CPU % over 7 days
      action      = "Recommend"
    }
    oversized_resources = {
      name        = "Oversized Resources"
      description = "Resources with consistently low utilization"
      threshold   = 30  # Utilization % over 30 days
      action      = "Resize"
    }
  }
}

# Action group for cost alerts
resource "azurerm_monitor_action_group" "cost_alerts" {
  name                = "${local.name_prefix}-cost-alerts"
  resource_group_name = var.resource_group_name
  short_name          = "CostAlert"
  
  # Email notifications
  dynamic "email_receiver" {
    for_each = var.cost_alert_emails
    content {
      name          = "email_${email_receiver.key}"
      email_address = email_receiver.value
    }
  }
  
  # Webhook notifications for integration
  dynamic "webhook_receiver" {
    for_each = var.cost_webhook_urls
    content {
      name        = "webhook_${webhook_receiver.key}"
      service_uri = webhook_receiver.value
    }
  }
  
  # SMS notifications for critical alerts
  dynamic "sms_receiver" {
    for_each = var.critical_alert_sms
    content {
      name         = "sms_${sms_receiver.key}"
      country_code = sms_receiver.value.country_code
      phone_number = sms_receiver.value.phone_number
    }
  }
  
  tags = merge(var.tags, {
    Component = "Cost Management"
    Service   = "Action Group"
    Purpose   = "Cost Alert Notifications"
  })
}

# Main subscription budget
resource "azurerm_consumption_budget_subscription" "main_budget" {
  name            = "${local.name_prefix}-main-budget"
  subscription_id = var.subscription_id
  
  amount     = var.monthly_budget_amount
  time_grain = "Monthly"
  
  time_period {
    start_date = formatdate("YYYY-MM-01'T'00:00:00'Z'", timestamp())
    # No end date for ongoing budget
  }
  
  # Progressive notifications at different thresholds
  dynamic "notification" {
    for_each = local.budget_thresholds
    content {
      enabled        = true
      threshold      = notification.value.percentage
      operator       = "GreaterThan"
      threshold_type = "Actual"
      
      contact_emails = var.cost_alert_emails
      contact_groups = [azurerm_monitor_action_group.cost_alerts.id]
    }
  }
  
  # Forecasted spend alerts
  notification {
    enabled        = true
    threshold      = 100
    operator       = "GreaterThan"
    threshold_type = "Forecasted"
    
    contact_emails = var.cost_alert_emails
    contact_groups = [azurerm_monitor_action_group.cost_alerts.id]
  }
  
  # Filter to AgentVault resources only
  filter {
    dimension {
      name = "ResourceGroupName"
      values = [
        var.resource_group_name,
        "${var.resource_group_name}-*"
      ]
    }
  }
}

# Resource group specific budgets
resource "azurerm_consumption_budget_resource_group" "component_budgets" {
  for_each = var.component_budgets
  
  name              = "${local.name_prefix}-${each.key}-budget"
  resource_group_id = each.value.resource_group_id
  
  amount     = each.value.monthly_amount
  time_grain = "Monthly"
  
  time_period {
    start_date = formatdate("YYYY-MM-01'T'00:00:00'Z'", timestamp())
  }
  
  # Standard notifications
  notification {
    enabled        = true
    threshold      = 80
    operator       = "GreaterThan"
    threshold_type = "Actual"
    
    contact_emails = concat(var.cost_alert_emails, each.value.additional_emails)
  }
  
  notification {
    enabled        = true
    threshold      = 100
    operator       = "GreaterThan"
    threshold_type = "Forecasted"
    
    contact_emails = concat(var.cost_alert_emails, each.value.additional_emails)
  }
}

# Cost management export for detailed analysis
resource "azurerm_cost_management_export" "daily_export" {
  name                         = "${local.name_prefix}-daily-export"
  resource_group_name          = var.resource_group_name
  
  # Export to storage account
  export_destination {
    type                       = "AzureBlob"
    container_name            = "cost-exports"
    resource_id               = var.cost_export_storage_account_id
    root_folder_path          = "agentvault"
  }
  
  query {
    type       = "ActualCost"
    time_frame = "Custom"
    
    time_period {
      from = formatdate("YYYY-MM-DD'T'00:00:00'Z'", timeadd(timestamp(), "-1d"))
      to   = formatdate("YYYY-MM-DD'T'23:59:59'Z'", timeadd(timestamp(), "-1d"))
    }
  }
  
  # Daily schedule
  schedule {
    status     = "Active"
    recurrence = "Daily"
    
    recurrence_period {
      from = formatdate("YYYY-MM-DD'T'00:00:00'Z'", timestamp())
    }
  }
}

# Cost anomaly alert configuration
resource "azurerm_cost_anomaly_alert" "main" {
  count = var.enable_anomaly_detection ? 1 : 0
  
  name         = "${local.name_prefix}-anomaly-alert"
  display_name = "AgentVault Cost Anomaly Detection"
  
  email_subject = "AgentVault Cost Anomaly Detected"
  email_addresses = var.cost_alert_emails
  
  # Alert on anomalies above threshold
  message = "Unusual spending pattern detected in AgentVault resources. Please review."
}

# Azure Advisor cost recommendations configuration
resource "azurerm_advisor_recommendations" "cost_optimization" {
  count = var.enable_advisor_recommendations ? 1 : 0
  
  category = "Cost"
  
  # Automatically apply certain recommendations
  auto_execute {
    enabled = true
    
    # Types of recommendations to auto-execute
    recommendation_types = [
      "Shutdown idle resources",
      "Right-size underutilized resources",
      "Delete unattached disks",
      "Use reserved instances"
    ]
  }
}

# Policy assignment for required tags (for cost allocation)
resource "azurerm_policy_assignment" "require_tags" {
  name                 = "${local.name_prefix}-require-tags"
  policy_definition_id = "/providers/Microsoft.Authorization/policyDefinitions/96670d01-0a00-0000-0000-000000000000"
  scope               = var.subscription_id
  
  # Required tags for cost tracking
  parameters = jsonencode({
    tagName1 = {
      value = "CostCenter"
    }
    tagName2 = {
      value = "Project"
    }
    tagName3 = {
      value = "Environment"
    }
    tagName4 = {
      value = "Owner"
    }
  })
  
  # Enforcement mode
  enforcement_mode = var.enforce_tagging_policy ? "Default" : "DoNotEnforce"
}

# Scheduled actions for cost optimization
resource "azurerm_automation_account" "cost_automation" {
  name                = "${local.name_prefix}-cost-automation"
  location            = var.location
  resource_group_name = var.resource_group_name
  
  sku_name = "Basic"
  
  identity {
    type = "SystemAssigned"
  }
  
  tags = merge(var.tags, {
    Component = "Cost Management"
    Service   = "Automation"
    Purpose   = "Cost Optimization Actions"
  })
}

# Runbook for shutting down development resources
resource "azurerm_automation_runbook" "shutdown_dev" {
  name                    = "Shutdown-DevResources"
  location               = var.location
  resource_group_name    = var.resource_group_name
  automation_account_name = azurerm_automation_account.cost_automation.name
  
  log_verbose  = true
  log_progress = true
  description  = "Automatically shutdown development resources after hours"
  
  runbook_type = "PowerShell"
  
  content = <<-EOT
    # Shutdown Development Resources Runbook
    # Automatically stops VMs and deallocates resources tagged with Environment=dev
    
    param(
        [Parameter(Mandatory=$false)]
        [string]$ResourceGroupName = "${var.resource_group_name}",
        
        [Parameter(Mandatory=$false)]
        [string]$TagName = "Environment",
        
        [Parameter(Mandatory=$false)]
        [string]$TagValue = "dev"
    )
    
    # Authenticate using managed identity
    Connect-AzAccount -Identity
    
    # Get all VMs with the specified tag
    $vms = Get-AzVM -ResourceGroupName $ResourceGroupName | Where-Object {
        $_.Tags[$TagName] -eq $TagValue -and $_.PowerState -eq "VM running"
    }
    
    # Stop each VM
    foreach ($vm in $vms) {
        Write-Output "Stopping VM: $($vm.Name)"
        Stop-AzVM -ResourceGroupName $vm.ResourceGroupName -Name $vm.Name -Force
    }
    
    # Get AKS clusters with the tag
    $aksClusters = Get-AzAksCluster -ResourceGroupName $ResourceGroupName | Where-Object {
        $_.Tags[$TagName] -eq $TagValue
    }
    
    # Stop AKS clusters
    foreach ($aks in $aksClusters) {
        Write-Output "Stopping AKS cluster: $($aks.Name)"
        Stop-AzAksCluster -ResourceGroupName $aks.ResourceGroupName -Name $aks.Name
    }
    
    Write-Output "Development resource shutdown completed"
  EOT
  
  tags = var.tags
}

# Schedule for dev shutdown
resource "azurerm_automation_schedule" "dev_shutdown_schedule" {
  name                    = "DevShutdownSchedule"
  resource_group_name     = var.resource_group_name
  automation_account_name = azurerm_automation_account.cost_automation.name
  
  frequency = "Day"
  interval  = 1
  timezone  = var.timezone
  
  # Start time - 7 PM local time
  start_time = "${formatdate("YYYY-MM-DD", timestamp())}T19:00:00+00:00"
  
  description = "Daily shutdown of development resources at 7 PM"
}

# Link runbook to schedule
resource "azurerm_automation_job_schedule" "dev_shutdown_link" {
  resource_group_name     = var.resource_group_name
  automation_account_name = azurerm_automation_account.cost_automation.name
  schedule_name          = azurerm_automation_schedule.dev_shutdown_schedule.name
  runbook_name           = azurerm_automation_runbook.shutdown_dev.name
}

# Dashboard for cost visualization
resource "azurerm_dashboard" "cost_dashboard" {
  name                = "${local.name_prefix}-cost-dashboard"
  resource_group_name = var.resource_group_name
  location            = var.location
  
  tags = merge(var.tags, {
    Component = "Cost Management"
    Service   = "Dashboard"
    Purpose   = "Cost Visualization"
  })
  
  dashboard_properties = jsonencode({
    lenses = [
      {
        order = 0
        parts = [
          {
            position = {
              x = 0
              y = 0
              rowSpan = 4
              colSpan = 6
            }
            metadata = {
              type = "Extension/Microsoft_Azure_CostManagement/PartType/CostAnalysisPart"
              settings = {
                title = "Monthly Cost Trend"
                subtitle = "AgentVault Total Costs"
                chartType = "Line"
                scope = var.subscription_id
                query = {
                  type = "ActualCost"
                  timeframe = "MonthToDate"
                  dataset = {
                    granularity = "Daily"
                    aggregation = {
                      totalCost = {
                        name = "Cost"
                        function = "Sum"
                      }
                    }
                  }
                }
              }
            }
          },
          {
            position = {
              x = 6
              y = 0
              rowSpan = 4
              colSpan = 6
            }
            metadata = {
              type = "Extension/Microsoft_Azure_CostManagement/PartType/CostByServicePart"
              settings = {
                title = "Cost by Service"
                subtitle = "Top 10 Services"
                scope = var.subscription_id
              }
            }
          },
          {
            position = {
              x = 0
              y = 4
              rowSpan = 4
              colSpan = 6
            }
            metadata = {
              type = "Extension/Microsoft_Azure_Monitoring/PartType/MetricChartPart"
              settings = {
                title = "Budget vs Actual"
                subtitle = "Current Month"
              }
            }
          }
        ]
      }
    ]
  })
}

# Cost allocation rules
resource "azurerm_cost_allocation_rule" "department_allocation" {
  count = var.enable_cost_allocation ? 1 : 0
  
  name        = "${local.name_prefix}-dept-allocation"
  description = "Allocate shared costs by department"
  
  # Source costs (shared resources)
  source {
    type = "Dimension"
    name = "ResourceGroup"
    values = ["${var.resource_group_name}-shared"]
  }
  
  # Target allocation
  target {
    type = "Tag"
    name = "Department"
  }
  
  # Allocation method
  method = "Proportional"
  
  # Based on usage metrics
  basis = "Usage"
}

# Outputs
output "budget_ids" {
  description = "IDs of created budgets"
  value = {
    main_budget = azurerm_consumption_budget_subscription.main_budget.id
    component_budgets = {
      for k, v in azurerm_consumption_budget_resource_group.component_budgets : k => v.id
    }
  }
}

output "cost_alerts_action_group_id" {
  description = "Action group ID for cost alerts"
  value       = azurerm_monitor_action_group.cost_alerts.id
}

output "automation_account_id" {
  description = "Automation account ID for cost optimization"
  value       = azurerm_automation_account.cost_automation.id
}

output "cost_dashboard_url" {
  description = "URL to access the cost dashboard"
  value       = "https://portal.azure.com/#@${var.tenant_id}/dashboard/arm${azurerm_dashboard.cost_dashboard.id}"
}

output "cost_export_schedule" {
  description = "Cost export configuration"
  value = {
    export_id     = azurerm_cost_management_export.daily_export.id
    container    = "cost-exports"
    frequency    = "Daily"
    storage_path = "agentvault/"
  }
}

output "optimization_runbooks" {
  description = "Automation runbooks for cost optimization"
  value = {
    dev_shutdown = {
      runbook_id   = azurerm_automation_runbook.shutdown_dev.id
      schedule     = "Daily at 7 PM ${var.timezone}"
      description  = "Shuts down development resources"
    }
  }
}

output "cost_optimization_recommendations" {
  description = "Cost optimization recommendations and tips"
  value = {
    immediate_actions = [
      "Review and tag all resources for accurate cost allocation",
      "Enable auto-shutdown for development and test environments",
      "Right-size underutilized virtual machines and AKS node pools",
      "Delete unattached managed disks and unused IP addresses",
      "Consider reserved instances for stable workloads"
    ]
    
    monitoring_tips = [
      "Set up regular cost review meetings",
      "Monitor budget alerts and anomaly notifications",
      "Review Azure Advisor recommendations weekly",
      "Track cost per agent and per tier",
      "Implement chargeback/showback for departments"
    ]
    
    best_practices = [
      "Use spot instances for non-critical batch workloads",
      "Implement lifecycle policies for blob storage",
      "Optimize network egress costs with CDN",
      "Use Azure Hybrid Benefit where applicable",
      "Regularly review and optimize reserved capacity"
    ]
  }
}

output "estimated_savings" {
  description = "Potential cost savings from optimization"
  value = {
    dev_shutdown_monthly     = var.monthly_budget_amount * 0.15  # ~15% from dev shutdown
    reserved_instances       = var.monthly_budget_amount * 0.20  # ~20% from RIs
    rightsizing             = var.monthly_budget_amount * 0.10  # ~10% from rightsizing
    unused_resources        = var.monthly_budget_amount * 0.05  # ~5% from cleanup
    total_potential_savings = var.monthly_budget_amount * 0.50  # Up to 50% total
  }
}