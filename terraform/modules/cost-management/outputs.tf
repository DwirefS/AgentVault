# AgentVault™ Cost Management Module - Outputs
# ===========================================
# Exported values for cost monitoring and optimization
#
# These outputs provide:
# - Budget and alert configuration details
# - Automation account information
# - Dashboard access URLs
# - Cost optimization recommendations
# - Estimated savings calculations
#
# Author: Dwiref Sharma
# Contact: DwirefS@SapientEdge.io

# Budget Information
output "budgets" {
  description = "Created budget resources with thresholds"
  value = {
    main_subscription = {
      id                = azurerm_consumption_budget_subscription.main_budget.id
      name              = azurerm_consumption_budget_subscription.main_budget.name
      amount            = azurerm_consumption_budget_subscription.main_budget.amount
      time_grain        = azurerm_consumption_budget_subscription.main_budget.time_grain
      
      # Alert thresholds
      thresholds = [
        for threshold in local.budget_thresholds : {
          percentage = threshold.percentage
          severity   = threshold.severity
          message    = threshold.message
        }
      ]
    }
    
    component_budgets = {
      for name, budget in azurerm_consumption_budget_resource_group.component_budgets : name => {
        id              = budget.id
        name            = budget.name
        amount          = budget.amount
        resource_group  = budget.resource_group_id
      }
    }
  }
}

# Alert Configuration
output "alert_configuration" {
  description = "Cost alert action group and notification settings"
  value = {
    action_group = {
      id         = azurerm_monitor_action_group.cost_alerts.id
      name       = azurerm_monitor_action_group.cost_alerts.name
      short_name = azurerm_monitor_action_group.cost_alerts.short_name
    }
    
    notification_channels = {
      emails   = var.cost_alert_emails
      webhooks = keys(var.cost_webhook_urls)
      sms      = keys(var.critical_alert_sms)
    }
    
    # Alert schedule
    alert_schedule = {
      budget_checks     = "Real-time"
      anomaly_detection = var.enable_anomaly_detection ? "Daily" : "Disabled"
      advisor_checks    = var.enable_advisor_recommendations ? "Weekly" : "Disabled"
    }
  }
}

# Automation Resources
output "automation" {
  description = "Cost optimization automation resources"
  value = {
    account = {
      id            = azurerm_automation_account.cost_automation.id
      name          = azurerm_automation_account.cost_automation.name
      identity_id   = azurerm_automation_account.cost_automation.identity[0].principal_id
    }
    
    runbooks = {
      dev_shutdown = {
        id          = azurerm_automation_runbook.shutdown_dev.id
        name        = azurerm_automation_runbook.shutdown_dev.name
        schedule    = azurerm_automation_schedule.dev_shutdown_schedule.frequency
        next_run    = azurerm_automation_schedule.dev_shutdown_schedule.start_time
        description = "Shuts down development resources daily at ${var.dev_shutdown_time}"
      }
    }
    
    schedules = {
      dev_shutdown = {
        enabled   = var.auto_shutdown_dev_resources
        time      = var.dev_shutdown_time
        timezone  = var.timezone
        frequency = "Daily (weekdays)"
      }
      
      weekend_scaling = {
        enabled   = var.auto_scale_weekends
        time      = "00:00"
        timezone  = var.timezone
        frequency = "Weekly (Saturday)"
      }
    }
  }
}

# Dashboard Access
output "dashboards" {
  description = "Cost visualization dashboard URLs"
  value = {
    main_dashboard = {
      id   = azurerm_dashboard.cost_dashboard.id
      name = azurerm_dashboard.cost_dashboard.name
      url  = "https://portal.azure.com/#@${var.tenant_id}/dashboard/arm${azurerm_dashboard.cost_dashboard.id}"
    }
    
    # Azure Cost Management URLs
    cost_analysis = {
      subscription = "https://portal.azure.com/#blade/Microsoft_Azure_CostManagement/Menu/costanalysis/scope/${urlencode(var.subscription_id)}"
      budgets      = "https://portal.azure.com/#blade/Microsoft_Azure_CostManagement/Menu/budgets"
      advisor      = "https://portal.azure.com/#blade/Microsoft_Azure_Expert/AdvisorMenuBlade/Cost"
    }
  }
}

# Cost Export Configuration
output "cost_exports" {
  description = "Cost export settings for detailed analysis"
  value = {
    daily_export = {
      id              = azurerm_cost_management_export.daily_export.id
      name            = azurerm_cost_management_export.daily_export.name
      storage_account = var.cost_export_storage_account_id
      container       = "cost-exports"
      path            = "agentvault/"
      schedule        = "Daily at midnight UTC"
      
      # Data location
      data_access = {
        storage_path = "wasbs://cost-exports@${split("/", var.cost_export_storage_account_id)[8]}.blob.core.windows.net/agentvault/"
        format      = "CSV"
        schema      = "Cost Management v1"
      }
    }
  }
}

# Optimization Insights
output "optimization_insights" {
  description = "Cost optimization recommendations and potential savings"
  value = {
    # Current settings
    current_optimizations = {
      dev_auto_shutdown        = var.auto_shutdown_dev_resources
      weekend_scaling          = var.auto_scale_weekends
      unused_resource_cleanup  = "${var.unused_resource_threshold_days} days"
      low_utilization_flagging = "${var.low_utilization_threshold_percent}%"
      anomaly_detection        = var.enable_anomaly_detection
      advisor_automation       = var.enable_advisor_recommendations
    }
    
    # Potential monthly savings
    estimated_savings_usd = {
      dev_shutdown     = var.auto_shutdown_dev_resources ? var.monthly_budget_amount * 0.15 : 0
      weekend_scaling  = var.auto_scale_weekends ? var.monthly_budget_amount * 0.05 : 0
      reserved_instances = var.enable_ri_recommendations ? var.monthly_budget_amount * 0.20 : 0
      rightsizing      = var.monthly_budget_amount * 0.10
      unused_cleanup   = var.monthly_budget_amount * 0.05
      
      total_potential = var.monthly_budget_amount * 0.30  # Conservative 30% estimate
    }
    
    # Quick wins
    immediate_actions = [
      "Enable all automation features for maximum savings",
      "Review and apply reserved instance recommendations",
      "Tag all resources with ${join(", ", var.cost_allocation_tags)}",
      "Set up weekly cost review meetings",
      "Configure department-level budgets"
    ]
  }
}

# Policy Configuration
output "policies" {
  description = "Cost-related policy assignments"
  value = {
    tagging_policy = {
      id               = azurerm_policy_assignment.require_tags.id
      name             = azurerm_policy_assignment.require_tags.name
      enforcement_mode = azurerm_policy_assignment.require_tags.enforcement_mode
      required_tags    = var.cost_allocation_tags
    }
    
    recommendations = var.enforce_tagging_policy ? [
      "Tagging policy is enforced - all new resources must have required tags",
      "Existing resources should be tagged retroactively",
      "Use 'DoNotEnforce' mode initially to identify non-compliant resources"
    ] : [
      "Enable tagging policy enforcement for better cost allocation",
      "Current mode allows resource creation without required tags"
    ]
  }
}

# Integration Instructions
output "integration_guide" {
  description = "Instructions for integrating cost management with applications"
  value = {
    webhook_integration = {
      format = "POST request with JSON payload"
      example_payload = jsonencode({
        alertType = "Budget"
        threshold = 90
        amount    = 9000
        budget    = 10000
        message   = "Budget usage at 90%"
      })
    }
    
    api_endpoints = {
      cost_query = "https://management.azure.com/subscriptions/${var.subscription_id}/providers/Microsoft.CostManagement/query?api-version=2021-10-01"
      budgets    = "https://management.azure.com/subscriptions/${var.subscription_id}/providers/Microsoft.Consumption/budgets?api-version=2021-10-01"
    }
    
    sdk_examples = {
      python = <<-EOT
        from azure.mgmt.costmanagement import CostManagementClient
        from azure.identity import DefaultAzureCredential
        
        credential = DefaultAzureCredential()
        client = CostManagementClient(credential, '${var.subscription_id}')
        
        # Query costs
        result = client.query.usage(
            scope=f'/subscriptions/${var.subscription_id}',
            parameters={
                'type': 'ActualCost',
                'timeframe': 'MonthToDate',
                'dataset': {
                    'granularity': 'Daily',
                    'aggregation': {
                        'totalCost': {'name': 'Cost', 'function': 'Sum'}
                    }
                }
            }
        )
      EOT
    }
  }
}

# Cost Allocation
output "cost_allocation" {
  description = "Cost allocation configuration for chargeback/showback"
  value = {
    enabled = var.enable_cost_allocation
    
    allocation_tags = var.cost_allocation_tags
    
    allocation_method = var.shared_resource_allocation_method
    
    reporting = {
      frequency  = var.report_frequency
      recipients = var.cost_report_recipients
      includes_recommendations = var.include_recommendations_in_report
    }
    
    # Sample allocation query
    sample_query = <<-EOT
      | where Tags contains "Department"
      | summarize TotalCost = sum(Cost) by tostring(Tags.Department)
      | order by TotalCost desc
    EOT
  }
}

# Summary Metrics
output "cost_management_summary" {
  description = "Summary of cost management configuration"
  value = {
    # Budget overview
    total_monthly_budget_usd = var.monthly_budget_amount + sum([for k, v in var.component_budgets : v.monthly_amount])
    
    # Monitoring coverage
    monitoring_coverage = {
      budget_alerts         = true
      anomaly_detection    = var.enable_anomaly_detection
      advisor_integration  = var.enable_advisor_recommendations
      automated_actions    = var.auto_shutdown_dev_resources || var.auto_scale_weekends
    }
    
    # Optimization potential
    optimization_score = sum([
      var.auto_shutdown_dev_resources ? 20 : 0,
      var.auto_scale_weekends ? 15 : 0,
      var.enable_anomaly_detection ? 15 : 0,
      var.enable_advisor_recommendations ? 20 : 0,
      var.enable_cost_allocation ? 15 : 0,
      var.enforce_tagging_policy ? 15 : 0
    ])  # Out of 100
    
    # Status
    status = "Configured and Active"
  }
}