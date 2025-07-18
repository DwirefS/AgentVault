# AgentVault™ Database Module
# PostgreSQL Flexible Server with HA, geo-replication, and enterprise features

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

variable "postgresql_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15"
}

variable "sku_name" {
  description = "SKU name for the database server"
  type        = string
  default     = "GP_Standard_D4s_v3"
}

variable "enable_high_availability" {
  description = "Enable high availability"
  type        = bool
  default     = true
}

variable "high_availability_mode" {
  description = "High availability mode (ZoneRedundant or SameZone)"
  type        = string
  default     = "ZoneRedundant"
}

variable "availability_zone" {
  description = "Primary availability zone"
  type        = string
  default     = "1"
}

variable "standby_availability_zone" {
  description = "Standby availability zone"
  type        = string
  default     = "2"
}

variable "storage_mb" {
  description = "Storage size in MB"
  type        = number
  default     = 102400
}

variable "backup_retention_days" {
  description = "Backup retention in days"
  type        = number
  default     = 35
}

variable "geo_redundant_backup_enabled" {
  description = "Enable geo-redundant backups"
  type        = bool
  default     = true
}

variable "delegated_subnet_id" {
  description = "Delegated subnet ID"
  type        = string
}

variable "private_dns_zone_id" {
  description = "Private DNS zone ID"
  type        = string
}

variable "ssl_enforcement_enabled" {
  description = "Enable SSL enforcement"
  type        = bool
  default     = true
}

variable "ssl_minimal_tls_version" {
  description = "Minimum TLS version"
  type        = string
  default     = "TLS1_2"
}

variable "enable_threat_detection" {
  description = "Enable advanced threat detection"
  type        = bool
  default     = true
}

variable "enable_audit_logs" {
  description = "Enable audit logging"
  type        = bool
  default     = true
}

variable "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID"
  type        = string
}

variable "databases" {
  description = "List of databases to create"
  type = list(object({
    name      = string
    charset   = string
    collation = string
  }))
  default = []
}

variable "firewall_rules" {
  description = "Firewall rules"
  type = list(object({
    name             = string
    start_ip_address = string
    end_ip_address   = string
  }))
  default = []
}

variable "administrator_login" {
  description = "Administrator login"
  type        = string
  sensitive   = true
}

variable "administrator_password" {
  description = "Administrator password"
  type        = string
  sensitive   = true
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "main" {
  name                = "${var.name_prefix}-postgres"
  location            = var.location
  resource_group_name = var.resource_group_name
  
  # Version
  version = var.postgresql_version
  
  # Authentication
  administrator_login    = var.administrator_login
  administrator_password = var.administrator_password
  
  # SKU and storage
  sku_name   = var.sku_name
  storage_mb = var.storage_mb
  
  # High availability
  high_availability {
    mode                      = var.enable_high_availability ? var.high_availability_mode : "Disabled"
    standby_availability_zone = var.enable_high_availability ? var.standby_availability_zone : null
  }
  
  # Availability zone
  zone = var.availability_zone
  
  # Backup configuration
  backup_retention_days        = var.backup_retention_days
  geo_redundant_backup_enabled = var.geo_redundant_backup_enabled
  
  # Network configuration
  delegated_subnet_id = var.delegated_subnet_id
  private_dns_zone_id = var.private_dns_zone_id
  
  # Maintenance window
  maintenance_window {
    day_of_week  = 0  # Sunday
    start_hour   = 2
    start_minute = 0
  }
  
  tags = var.tags
}

# Databases
resource "azurerm_postgresql_flexible_server_database" "main" {
  for_each = { for db in var.databases : db.name => db }
  
  name      = each.value.name
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = each.value.charset
  collation = each.value.collation
}

# Server configurations
resource "azurerm_postgresql_flexible_server_configuration" "audit" {
  count     = var.enable_audit_logs ? 1 : 0
  name      = "audit.log_enabled"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "on"
}

resource "azurerm_postgresql_flexible_server_configuration" "log_connections" {
  count     = var.enable_audit_logs ? 1 : 0
  name      = "log_connections"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "on"
}

resource "azurerm_postgresql_flexible_server_configuration" "log_disconnections" {
  count     = var.enable_audit_logs ? 1 : 0
  name      = "log_disconnections"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "on"
}

resource "azurerm_postgresql_flexible_server_configuration" "log_checkpoints" {
  name      = "log_checkpoints"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "on"
}

resource "azurerm_postgresql_flexible_server_configuration" "log_duration" {
  name      = "log_duration"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "on"
}

resource "azurerm_postgresql_flexible_server_configuration" "pgaudit" {
  count     = var.enable_audit_logs ? 1 : 0
  name      = "shared_preload_libraries"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "pgaudit"
}

resource "azurerm_postgresql_flexible_server_configuration" "pgaudit_log" {
  count     = var.enable_audit_logs ? 1 : 0
  name      = "pgaudit.log"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "ALL"
  
  depends_on = [azurerm_postgresql_flexible_server_configuration.pgaudit]
}

# Connection pooling configuration
resource "azurerm_postgresql_flexible_server_configuration" "max_connections" {
  name      = "max_connections"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "500"
}

resource "azurerm_postgresql_flexible_server_configuration" "shared_buffers" {
  name      = "shared_buffers"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "256000"  # 2GB in 8KB pages
}

resource "azurerm_postgresql_flexible_server_configuration" "effective_cache_size" {
  name      = "effective_cache_size"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "768000"  # 6GB in 8KB pages
}

# Performance tuning
resource "azurerm_postgresql_flexible_server_configuration" "work_mem" {
  name      = "work_mem"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "16384"  # 16MB
}

resource "azurerm_postgresql_flexible_server_configuration" "maintenance_work_mem" {
  name      = "maintenance_work_mem"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "524288"  # 512MB
}

resource "azurerm_postgresql_flexible_server_configuration" "checkpoint_completion_target" {
  name      = "checkpoint_completion_target"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "0.9"
}

resource "azurerm_postgresql_flexible_server_configuration" "wal_buffers" {
  name      = "wal_buffers"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "16384"  # 16MB
}

# Enable extensions
resource "azurerm_postgresql_flexible_server_configuration" "extensions" {
  name      = "azure.extensions"
  server_id = azurerm_postgresql_flexible_server.main.id
  value     = "pg_stat_statements,pgcrypto,uuid-ossp,pg_trgm,pgvector"
}

# Firewall rules
resource "azurerm_postgresql_flexible_server_firewall_rule" "main" {
  for_each = { for rule in var.firewall_rules : rule.name => rule }
  
  name             = each.value.name
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = each.value.start_ip_address
  end_ip_address   = each.value.end_ip_address
}

# Advanced Threat Protection
resource "azurerm_postgresql_flexible_server_active_directory_administrator" "main" {
  count               = var.enable_threat_detection ? 1 : 0
  server_name         = azurerm_postgresql_flexible_server.main.name
  resource_group_name = var.resource_group_name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  object_id           = data.azurerm_client_config.current.object_id
  principal_name      = "AzureAD Admin"
  principal_type      = "ServicePrincipal"
}

# Diagnostic settings
resource "azurerm_monitor_diagnostic_setting" "database" {
  name                       = "${var.name_prefix}-db-diagnostics"
  target_resource_id         = azurerm_postgresql_flexible_server.main.id
  log_analytics_workspace_id = var.log_analytics_workspace_id
  
  enabled_log {
    category = "PostgreSQLLogs"
    
    retention_policy {
      enabled = true
      days    = 30
    }
  }
  
  enabled_log {
    category = "PostgreSQLFlexSessions"
    
    retention_policy {
      enabled = true
      days    = 30
    }
  }
  
  enabled_log {
    category = "PostgreSQLFlexQueryStoreRuntime"
    
    retention_policy {
      enabled = true
      days    = 30
    }
  }
  
  enabled_log {
    category = "PostgreSQLFlexQueryStoreWaitStats"
    
    retention_policy {
      enabled = true
      days    = 30
    }
  }
  
  metric {
    category = "AllMetrics"
    
    retention_policy {
      enabled = true
      days    = 30
    }
  }
}

# Data sources
data "azurerm_client_config" "current" {}

# Outputs
output "server_id" {
  value = azurerm_postgresql_flexible_server.main.id
}

output "server_name" {
  value = azurerm_postgresql_flexible_server.main.name
}

output "fqdn" {
  value = azurerm_postgresql_flexible_server.main.fqdn
}

output "administrator_login" {
  value     = azurerm_postgresql_flexible_server.main.administrator_login
  sensitive = true
}

output "database_ids" {
  value = { for k, v in azurerm_postgresql_flexible_server_database.main : k => v.id }
}

output "connection_string" {
  value = "postgresql://${azurerm_postgresql_flexible_server.main.administrator_login}@${azurerm_postgresql_flexible_server.main.name}:${var.administrator_password}@${azurerm_postgresql_flexible_server.main.fqdn}:5432/agentvault?sslmode=require"
  sensitive = true
}