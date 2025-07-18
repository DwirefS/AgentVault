# AgentVault™ Redis Cache Module - Input Variables
# ================================================
# Configuration variables for the distributed cache infrastructure
#
# These variables control:
# - Cache tier configurations and sizing
# - Network and security settings
# - High availability and disaster recovery
# - Monitoring and alerting thresholds
# - Performance optimization parameters
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
  description = "Azure region for Redis deployment"
  type        = string
}

variable "resource_group_name" {
  description = "Resource group name for Redis resources"
  type        = string
}

variable "tags" {
  description = "Tags to apply to all Redis resources"
  type        = map(string)
  default     = {}
}

# Networking Configuration
variable "virtual_network_id" {
  description = "Virtual network ID for DNS zone linking"
  type        = string
}

variable "cache_subnet_id" {
  description = "Subnet ID for Redis cache instances (Premium SKU only)"
  type        = string
  default     = null
}

variable "private_endpoint_subnet_id" {
  description = "Subnet ID for private endpoints"
  type        = string
  default     = null
}

variable "aks_subnet_start_ip" {
  description = "Start IP of AKS subnet for firewall rules"
  type        = string
  default     = null
}

variable "aks_subnet_end_ip" {
  description = "End IP of AKS subnet for firewall rules"
  type        = string
  default     = null
}

# Security Configuration
variable "key_vault_id" {
  description = "Key Vault ID for storing Redis passwords"
  type        = string
}

variable "enable_private_endpoint" {
  description = "Enable private endpoints for Redis instances"
  type        = bool
  default     = true
}

# High Availability Configuration
variable "enable_zone_redundancy" {
  description = "Enable zone redundancy for Redis instances"
  type        = bool
  default     = true
}

variable "enable_redis_enterprise" {
  description = "Enable Redis Enterprise for ultra-low latency"
  type        = bool
  default     = false
}

# Backup Configuration
variable "backup_storage_connection_string" {
  description = "Storage account connection string for Redis backups"
  type        = string
  sensitive   = true
  default     = null
}

# Monitoring Configuration
variable "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID for diagnostics"
  type        = string
}

variable "action_group_id" {
  description = "Action group ID for alert notifications"
  type        = string
}

# Performance Configuration
variable "redis_cache_sizes" {
  description = "Override default cache sizes for each tier"
  type = map(object({
    capacity    = number
    shard_count = number
  }))
  default = {}
}

variable "redis_performance_settings" {
  description = "Performance tuning settings for Redis"
  type = object({
    maxmemory_reserved_percentage = number
    maxmemory_delta_percentage    = number
    tcp_keepalive_seconds        = number
    timeout_seconds              = number
  })
  default = {
    maxmemory_reserved_percentage = 25
    maxmemory_delta_percentage    = 25
    tcp_keepalive_seconds        = 60
    timeout_seconds              = 300
  }
}

# Alert Thresholds
variable "alert_thresholds" {
  description = "Thresholds for Redis monitoring alerts"
  type = object({
    memory_usage_percentage = number
    cpu_usage_percentage    = number
    connection_count        = number
    cache_miss_rate        = number
  })
  default = {
    memory_usage_percentage = 90
    cpu_usage_percentage    = 80
    connection_count        = 10000
    cache_miss_rate        = 25
  }
}

# Circuit Breaker Configuration
variable "circuit_breaker_settings" {
  description = "Circuit breaker configuration for fault tolerance"
  type = object({
    enabled                = bool
    failure_threshold      = number
    success_threshold      = number
    timeout_seconds        = number
    half_open_max_requests = number
  })
  default = {
    enabled                = true
    failure_threshold      = 5
    success_threshold      = 2
    timeout_seconds        = 30
    half_open_max_requests = 3
  }
}

# Feature Flags
variable "enable_redis_modules" {
  description = "Enable Redis modules (RediSearch, RedisJSON, etc.)"
  type        = bool
  default     = true
}

variable "enable_geo_replication" {
  description = "Enable geo-replication for disaster recovery"
  type        = bool
  default     = false
}

variable "geo_replication_regions" {
  description = "Regions for geo-replication"
  type        = list(string)
  default     = []
}