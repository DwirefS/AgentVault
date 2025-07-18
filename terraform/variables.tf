# AgentVault™ Terraform Variables
# Configuration variables for enterprise AI agent storage platform
#
# Author: Dwiref Sharma
# Contact: DwirefS@SapientEdge.io

# Environment Configuration
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "location" {
  description = "Azure region for resource deployment"
  type        = string
  default     = "East US 2"
  validation {
    condition = contains([
      "East US", "East US 2", "West US 2", "West US 3",
      "Central US", "North Central US", "South Central US",
      "West Central US", "Canada Central", "Canada East",
      "North Europe", "West Europe", "UK South", "UK West",
      "France Central", "Germany West Central", "Switzerland North",
      "Norway East", "Sweden Central", "Italy North",
      "Poland Central", "Spain Central", "Israel Central",
      "UAE North", "South Africa North", "Australia East",
      "Australia Southeast", "Southeast Asia", "East Asia",
      "Japan East", "Japan West", "Korea Central", "Korea South",
      "Central India", "South India", "West India", "Jio India West",
      "Brazil South", "Brazil Southeast"
    ], var.location)
    error_message = "Location must be a valid Azure region."
  }
}

# Network Configuration
variable "allowed_ip_ranges" {
  description = "List of IP ranges allowed to access AgentVault resources"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production
  validation {
    condition     = length(var.allowed_ip_ranges) > 0
    error_message = "At least one IP range must be specified."
  }
}

variable "enable_bastion" {
  description = "Enable Azure Bastion for secure VM access"
  type        = bool
  default     = true
}

# Azure NetApp Files Configuration
variable "anf_capacity_pools" {
  description = "Configuration for ANF capacity pools by performance tier"
  type = map(object({
    service_level = string
    size_gb      = number
    qos_type     = string
  }))
  default = {
    ultra = {
      service_level = "Ultra"
      size_gb      = 4096  # 4TB minimum for Ultra
      qos_type     = "Auto"
    }
    premium = {
      service_level = "Premium"
      size_gb      = 8192  # 8TB for premium workloads
      qos_type     = "Auto"
    }
    standard = {
      service_level = "Standard"
      size_gb      = 16384  # 16TB for standard workloads
      qos_type     = "Auto"
    }
  }
}

variable "anf_volumes" {
  description = "Initial ANF volumes to create"
  type = map(object({
    capacity_pool    = string
    size_gb         = number
    protocol_types  = list(string)
    export_policy   = map(any)
    snapshot_policy = bool
    backup_enabled  = bool
  }))
  default = {
    "vector-store" = {
      capacity_pool   = "ultra"
      size_gb        = 1024  # 1TB for vector storage
      protocol_types = ["NFSv4.1"]
      export_policy = {
        rule_index         = 1
        allowed_clients    = "10.0.0.0/16"
        unix_read_write    = true
        root_access        = false
        kerberos_enabled   = false
      }
      snapshot_policy = true
      backup_enabled  = true
    }
    "memory-cache" = {
      capacity_pool   = "premium"
      size_gb        = 2048  # 2TB for memory cache
      protocol_types = ["NFSv4.1"]
      export_policy = {
        rule_index         = 1
        allowed_clients    = "10.0.0.0/16"
        unix_read_write    = true
        root_access        = false
        kerberos_enabled   = false
      }
      snapshot_policy = true
      backup_enabled  = true
    }
    "knowledge-base" = {
      capacity_pool   = "standard"
      size_gb        = 4096  # 4TB for knowledge base
      protocol_types = ["NFSv4.1"]
      export_policy = {
        rule_index         = 1
        allowed_clients    = "10.0.0.0/16"
        unix_read_write    = true
        root_access        = false
        kerberos_enabled   = false
      }
      snapshot_policy = true
      backup_enabled  = true
    }
  }
}

# Redis Cache Configuration
variable "redis_capacity" {
  description = "Redis cache capacity"
  type        = number
  default     = 6
  validation {
    condition     = contains([0, 1, 2, 3, 4, 5, 6], var.redis_capacity)
    error_message = "Redis capacity must be between 0-6."
  }
}

variable "redis_family" {
  description = "Redis cache family"
  type        = string
  default     = "P"
  validation {
    condition     = contains(["C", "P"], var.redis_family)
    error_message = "Redis family must be C (Basic/Standard) or P (Premium)."
  }
}

variable "redis_sku" {
  description = "Redis cache SKU"
  type        = string
  default     = "Premium"
  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.redis_sku)
    error_message = "Redis SKU must be Basic, Standard, or Premium."
  }
}

variable "redis_zones" {
  description = "Availability zones for Redis cache"
  type        = list(string)
  default     = ["1", "2", "3"]
}

# Compute Configuration
variable "enable_aks" {
  description = "Enable Azure Kubernetes Service for container workloads"
  type        = bool
  default     = true
}

variable "aks_node_count" {
  description = "Initial number of AKS nodes"
  type        = number
  default     = 3
  validation {
    condition     = var.aks_node_count >= 1 && var.aks_node_count <= 100
    error_message = "AKS node count must be between 1 and 100."
  }
}

variable "aks_node_vm_size" {
  description = "VM size for AKS nodes"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "enable_container_apps" {
  description = "Enable Azure Container Apps for serverless containers"
  type        = bool
  default     = true
}

variable "enable_functions" {
  description = "Enable Azure Functions for serverless compute"
  type        = bool
  default     = true
}

# Security Configuration
variable "enable_private_endpoints" {
  description = "Enable private endpoints for enhanced security"
  type        = bool
  default     = true
}

variable "key_vault_sku" {
  description = "Key Vault SKU"
  type        = string
  default     = "premium"
  validation {
    condition     = contains(["standard", "premium"], var.key_vault_sku)
    error_message = "Key Vault SKU must be standard or premium."
  }
}

variable "enable_disk_encryption" {
  description = "Enable disk encryption at rest"
  type        = bool
  default     = true
}

variable "enable_network_encryption" {
  description = "Enable network encryption in transit"
  type        = bool
  default     = true
}

# Monitoring Configuration
variable "log_retention_days" {
  description = "Log retention period in days"
  type        = number
  default     = 90
  validation {
    condition     = var.log_retention_days >= 30 && var.log_retention_days <= 730
    error_message = "Log retention must be between 30 and 730 days."
  }
}

variable "enable_alerting" {
  description = "Enable monitoring alerts"
  type        = bool
  default     = true
}

variable "alert_email" {
  description = "Email address for alerts"
  type        = string
  default     = "DwirefS@SapientEdge.io"
  validation {
    condition     = can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", var.alert_email))
    error_message = "Alert email must be a valid email address."
  }
}

# Backup Configuration
variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 365
  validation {
    condition     = var.backup_retention_days >= 7 && var.backup_retention_days <= 2555
    error_message = "Backup retention must be between 7 and 2555 days."
  }
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup for disaster recovery"
  type        = bool
  default     = true
}

variable "backup_frequency" {
  description = "Backup frequency (Daily, Weekly)"
  type        = string
  default     = "Daily"
  validation {
    condition     = contains(["Daily", "Weekly"], var.backup_frequency)
    error_message = "Backup frequency must be Daily or Weekly."
  }
}

# Performance Configuration
variable "enable_accelerated_networking" {
  description = "Enable accelerated networking for improved performance"
  type        = bool
  default     = true
}

variable "enable_proximity_placement" {
  description = "Enable proximity placement groups for low latency"
  type        = bool
  default     = true
}

# Cost Optimization
variable "enable_auto_scaling" {
  description = "Enable auto-scaling for cost optimization"
  type        = bool
  default     = true
}

variable "enable_spot_instances" {
  description = "Enable spot instances for non-critical workloads"
  type        = bool
  default     = false  # Disabled by default for production
}

variable "schedule_shutdown" {
  description = "Enable scheduled shutdown for development environments"
  type        = bool
  default     = false
}

# Compliance Configuration
variable "enable_compliance_monitoring" {
  description = "Enable compliance monitoring and reporting"
  type        = bool
  default     = true
}

variable "compliance_standards" {
  description = "List of compliance standards to enforce"
  type        = list(string)
  default     = ["SOC2", "GDPR", "HIPAA", "PCI-DSS"]
}

variable "data_residency_region" {
  description = "Data residency region for compliance"
  type        = string
  default     = ""
}

# AI/ML Configuration
variable "enable_cognitive_services" {
  description = "Enable Azure Cognitive Services integration"
  type        = bool
  default     = true
}

variable "openai_deployment" {
  description = "Azure OpenAI deployment configuration"
  type = object({
    enabled = bool
    models  = list(string)
    sku     = string
  })
  default = {
    enabled = true
    models  = ["gpt-4", "gpt-35-turbo", "text-embedding-ada-002"]
    sku     = "S0"
  }
}

# Tags
variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "AgentVault™"
    Owner       = "Dwiref Sharma"
    Contact     = "DwirefS@SapientEdge.io"
    Repository  = "https://github.com/DwirefS/AgentVault"
    License     = "MIT"
    Purpose     = "Enterprise AI Agent Storage Platform"
  }
}

# Feature Flags
variable "feature_flags" {
  description = "Feature flags for experimental features"
  type = object({
    neural_compression     = bool
    time_travel_debugging  = bool
    cognitive_load_balancing = bool
    storage_dna_profiling  = bool
    memory_marketplace     = bool
    quantum_ready_security = bool
  })
  default = {
    neural_compression       = true
    time_travel_debugging    = true
    cognitive_load_balancing = true
    storage_dna_profiling    = true
    memory_marketplace       = false  # Beta feature
    quantum_ready_security   = false  # Future feature
  }
}