# AgentVault™ Cost Management Module - Input Variables
# ===================================================
# Configuration variables for cost optimization and budget control
#
# These variables control:
# - Budget amounts and thresholds
# - Alert recipients and notification channels
# - Cost optimization automation settings
# - Tagging and allocation policies
# - Export and reporting configuration
#
# Author: Dwiref Sharma
# Contact: DwirefS@SapientEdge.io

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "location" {
  description = "Azure region for cost management resources"
  type        = string
}

variable "resource_group_name" {
  description = "Resource group name for cost management resources"
  type        = string
}

variable "subscription_id" {
  description = "Azure subscription ID for budget scope"
  type        = string
}

variable "tenant_id" {
  description = "Azure AD tenant ID for dashboard URLs"
  type        = string
}

variable "tags" {
  description = "Tags to apply to all cost management resources"
  type        = map(string)
  default     = {}
}

# Budget Configuration
variable "monthly_budget_amount" {
  description = "Monthly budget amount in USD for the main subscription"
  type        = number
  validation {
    condition     = var.monthly_budget_amount > 0
    error_message = "Monthly budget must be greater than 0."
  }
}

variable "component_budgets" {
  description = "Budgets for individual components/resource groups"
  type = map(object({
    resource_group_id = string
    monthly_amount    = number
    additional_emails = list(string)
  }))
  default = {}
}

# Alert Configuration
variable "cost_alert_emails" {
  description = "Email addresses for cost alerts"
  type        = list(string)
  validation {
    condition     = length(var.cost_alert_emails) > 0
    error_message = "At least one email address must be provided for cost alerts."
  }
}

variable "cost_webhook_urls" {
  description = "Webhook URLs for cost alert integration (e.g., Slack, Teams)"
  type        = map(string)
  default     = {}
}

variable "critical_alert_sms" {
  description = "SMS numbers for critical cost alerts"
  type = map(object({
    country_code = string
    phone_number = string
  }))
  default = {}
}

# Cost Export Configuration
variable "cost_export_storage_account_id" {
  description = "Storage account ID for cost export data"
  type        = string
}

# Automation Configuration
variable "timezone" {
  description = "Timezone for scheduled actions (e.g., 'Pacific Standard Time')"
  type        = string
  default     = "UTC"
}

variable "dev_shutdown_time" {
  description = "Time to shutdown development resources (24-hour format)"
  type        = string
  default     = "19:00"
}

variable "dev_startup_time" {
  description = "Time to start development resources (24-hour format)"
  type        = string
  default     = "07:00"
}

# Feature Flags
variable "enable_anomaly_detection" {
  description = "Enable cost anomaly detection and alerts"
  type        = bool
  default     = true
}

variable "enable_advisor_recommendations" {
  description = "Enable and auto-apply Azure Advisor cost recommendations"
  type        = bool
  default     = true
}

variable "enable_cost_allocation" {
  description = "Enable cost allocation rules for chargeback/showback"
  type        = bool
  default     = true
}

variable "enforce_tagging_policy" {
  description = "Enforce required tags policy (blocks resource creation without tags)"
  type        = bool
  default     = false
}

# Optimization Settings
variable "auto_shutdown_dev_resources" {
  description = "Automatically shutdown development resources after hours"
  type        = bool
  default     = true
}

variable "auto_scale_weekends" {
  description = "Automatically scale down resources on weekends"
  type        = bool
  default     = true
}

variable "unused_resource_threshold_days" {
  description = "Days before flagging resources as unused"
  type        = number
  default     = 7
}

variable "low_utilization_threshold_percent" {
  description = "CPU utilization percentage to flag resources as underutilized"
  type        = number
  default     = 10
}

# Reserved Instance Configuration
variable "enable_ri_recommendations" {
  description = "Enable reserved instance purchase recommendations"
  type        = bool
  default     = true
}

variable "ri_recommendation_lookback_days" {
  description = "Days to analyze for RI recommendations"
  type        = number
  default     = 30
}

variable "ri_coverage_target_percent" {
  description = "Target percentage for reserved instance coverage"
  type        = number
  default     = 80
}

# Cost Anomaly Settings
variable "anomaly_detection_sensitivity" {
  description = "Sensitivity for anomaly detection (Low, Medium, High)"
  type        = string
  default     = "Medium"
  validation {
    condition     = contains(["Low", "Medium", "High"], var.anomaly_detection_sensitivity)
    error_message = "Sensitivity must be Low, Medium, or High."
  }
}

variable "anomaly_threshold_percent" {
  description = "Percentage increase to trigger anomaly alert"
  type        = number
  default     = 20
}

# Department/Project Allocation
variable "cost_allocation_tags" {
  description = "Tags to use for cost allocation"
  type        = list(string)
  default     = ["Department", "Project", "Owner", "CostCenter"]
}

variable "shared_resource_allocation_method" {
  description = "Method for allocating shared resource costs (Proportional, Even, Custom)"
  type        = string
  default     = "Proportional"
}

# Reporting Configuration
variable "cost_report_recipients" {
  description = "Email addresses for scheduled cost reports"
  type        = list(string)
  default     = []
}

variable "report_frequency" {
  description = "Frequency of cost reports (Daily, Weekly, Monthly)"
  type        = string
  default     = "Weekly"
}

variable "include_recommendations_in_report" {
  description = "Include optimization recommendations in cost reports"
  type        = bool
  default     = true
}