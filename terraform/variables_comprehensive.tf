# AgentVault™ - Terraform Variables
# Complete variable definitions for enterprise deployment

# Core Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "agentvault"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "location" {
  description = "Primary Azure region for deployment"
  type        = string
  default     = "eastus"
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Business Configuration
variable "cost_center" {
  description = "Cost center for billing"
  type        = string
}

variable "business_unit" {
  description = "Business unit owning the resources"
  type        = string
}

variable "monthly_budget" {
  description = "Monthly budget in USD"
  type        = number
  default     = 10000
}

variable "budget_alert_emails" {
  description = "Email addresses for budget alerts"
  type        = list(string)
  default     = []
}

# Networking Configuration
variable "vnet_address_space" {
  description = "Address space for the virtual network"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "enable_azure_firewall" {
  description = "Enable Azure Firewall for network security"
  type        = bool
  default     = true
}

variable "enable_bastion" {
  description = "Enable Azure Bastion for secure access"
  type        = bool
  default     = true
}

variable "allowed_ips" {
  description = "List of allowed IP addresses for access"
  type        = list(string)
  default     = []
}

# Security Configuration
variable "enable_hsm" {
  description = "Enable Hardware Security Module for key management"
  type        = bool
  default     = false
}

variable "admin_object_ids" {
  description = "Azure AD object IDs for administrators"
  type        = list(string)
}

variable "domain_name" {
  description = "Custom domain name for the application"
  type        = string
}

# AKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for AKS"
  type        = string
  default     = "1.28.3"
}

variable "aks_system_node_size" {
  description = "VM size for system node pool"
  type        = string
  default     = "Standard_D4s_v5"
}

variable "aks_agent_node_count" {
  description = "Number of agent nodes"
  type        = number
  default     = 3
}

variable "aks_agent_max_count" {
  description = "Maximum number of agent nodes for autoscaling"
  type        = number
  default     = 10
}

variable "aks_agent_node_size" {
  description = "VM size for agent node pool"
  type        = string
  default     = "Standard_NC6s_v3" # GPU-enabled for AI workloads
}

variable "aks_monitoring_node_size" {
  description = "VM size for monitoring node pool"
  type        = string
  default     = "Standard_D4s_v5"
}

# ANF Configuration
variable "anf_premium_pool_size" {
  description = "Size of premium capacity pool in TB"
  type        = number
  default     = 4
}

variable "anf_standard_pool_size" {
  description = "Size of standard capacity pool in TB"
  type        = number
  default     = 8
}

variable "anf_ultra_pool_size" {
  description = "Size of ultra capacity pool in TB"
  type        = number
  default     = 2
}

variable "anf_agents_volume_size" {
  description = "Size of agents volume in TB"
  type        = number
  default     = 2
}

variable "anf_models_volume_size" {
  description = "Size of models volume in TB"
  type        = number
  default     = 4
}

variable "anf_artifacts_volume_size" {
  description = "Size of artifacts volume in TB"
  type        = number
  default     = 2
}

# Active Directory Configuration
variable "ad_dns_servers" {
  description = "DNS servers for Active Directory"
  type        = list(string)
  default     = []
}

variable "ad_domain" {
  description = "Active Directory domain name"
  type        = string
  default     = ""
}

variable "ad_username" {
  description = "Active Directory admin username"
  type        = string
  default     = ""
  sensitive   = true
}

variable "ad_password" {
  description = "Active Directory admin password"
  type        = string
  default     = ""
  sensitive   = true
}

# Database Configuration
variable "database_storage_mb" {
  description = "Database storage size in MB"
  type        = number
  default     = 102400 # 100GB
}

variable "database_admin_username" {
  description = "Database administrator username"
  type        = string
  default     = "agentvault_admin"
  sensitive   = true
}

# Monitoring Configuration
variable "alert_email_addresses" {
  description = "Email addresses for alerts"
  type        = list(string)
  default     = []
}

variable "alert_sms_numbers" {
  description = "SMS numbers for critical alerts"
  type        = list(string)
  default     = []
}

variable "alert_webhooks" {
  description = "Webhook URLs for alerts"
  type        = list(string)
  default     = []
}

# Disaster Recovery Configuration
variable "enable_disaster_recovery" {
  description = "Enable disaster recovery features"
  type        = bool
  default     = true
}

# Feature Flags
variable "enable_autoscaling" {
  description = "Enable autoscaling for all applicable resources"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable comprehensive monitoring"
  type        = bool
  default     = true
}

variable "enable_backup" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "enable_encryption" {
  description = "Enable encryption at rest and in transit"
  type        = bool
  default     = true
}

# Compliance Configuration
variable "compliance_standards" {
  description = "List of compliance standards to enforce"
  type        = list(string)
  default     = ["SOC2", "ISO27001", "HIPAA"]
}

variable "data_retention_days" {
  description = "Number of days to retain data"
  type        = number
  default     = 90
}

# Performance Configuration
variable "enable_ultra_performance" {
  description = "Enable ultra performance tier for critical workloads"
  type        = bool
  default     = false
}

variable "cache_size_gb" {
  description = "Size of distributed cache in GB"
  type        = number
  default     = 64
}

# Development/Testing Configuration
variable "enable_dev_tools" {
  description = "Enable development tools (non-production only)"
  type        = bool
  default     = false
}

variable "enable_debug_logging" {
  description = "Enable debug logging (non-production only)"
  type        = bool
  default     = false
}

# Cost Management Configuration
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

variable "cost_report_additional_emails" {
  description = "Additional email addresses for cost reports"
  type        = list(string)
  default     = []
}

variable "timezone" {
  description = "Timezone for scheduled actions (e.g., 'Pacific Standard Time')"
  type        = string
  default     = "UTC"
}