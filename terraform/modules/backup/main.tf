# AgentVault™ Backup Module
# Recovery Services Vault with automated backup policies and geo-redundancy

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

variable "vault_sku" {
  description = "Recovery Services Vault SKU"
  type        = string
  default     = "Standard"
}

variable "backup_policies" {
  description = "List of backup policies"
  type = list(object({
    name              = string
    timezone          = string
    backup_frequency  = string
    backup_time       = string
    backup_weekdays   = optional(list(string), [])
    retention_daily   = number
    retention_weekly  = number
    retention_monthly = number
    retention_yearly  = number
  }))
  default = []
}

variable "enable_vm_backup" {
  description = "Enable VM backup"
  type        = bool
  default     = true
}

variable "enable_file_backup" {
  description = "Enable file backup"
  type        = bool
  default     = true
}

variable "enable_database_backup" {
  description = "Enable database backup"
  type        = bool
  default     = true
}

variable "storage_type" {
  description = "Storage type (LocallyRedundant or GeoRedundant)"
  type        = string
  default     = "GeoRedundant"
}

variable "soft_delete_enabled" {
  description = "Enable soft delete"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Recovery Services Vault
resource "azurerm_recovery_services_vault" "main" {
  name                = "${var.name_prefix}-rsv"
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = var.vault_sku
  
  # Storage settings
  storage_mode_type    = var.storage_type
  soft_delete_enabled  = var.soft_delete_enabled
  
  # Immutability
  immutability = "Unlocked"
  
  # Identity for encryption
  identity {
    type = "SystemAssigned"
  }
  
  # Monitoring
  monitoring {
    alerts_for_all_job_failures_enabled            = true
    alerts_for_critical_operation_failures_enabled = true
  }
  
  tags = var.tags
}

# Backup Policies
resource "azurerm_backup_policy_vm" "main" {
  for_each            = { for policy in var.backup_policies : policy.name => policy if var.enable_vm_backup }
  name                = each.value.name
  resource_group_name = var.resource_group_name
  recovery_vault_name = azurerm_recovery_services_vault.main.name
  
  timezone = each.value.timezone
  
  backup {
    frequency = each.value.backup_frequency
    time      = each.value.backup_time
    weekdays  = each.value.backup_frequency == "Weekly" ? each.value.backup_weekdays : null
  }
  
  retention_daily {
    count = each.value.retention_daily
  }
  
  dynamic "retention_weekly" {
    for_each = each.value.retention_weekly > 0 ? [1] : []
    content {
      count    = each.value.retention_weekly
      weekdays = ["Sunday", "Wednesday", "Friday"]
    }
  }
  
  dynamic "retention_monthly" {
    for_each = each.value.retention_monthly > 0 ? [1] : []
    content {
      count    = each.value.retention_monthly
      weekdays = ["Sunday"]
      weeks    = ["First", "Last"]
    }
  }
  
  dynamic "retention_yearly" {
    for_each = each.value.retention_yearly > 0 ? [1] : []
    content {
      count    = each.value.retention_yearly
      weekdays = ["Sunday"]
      weeks    = ["Last"]
      months   = ["January", "July"]
    }
  }
  
  # Instant restore
  instant_restore_retention_days = 5
  
  # Tiering policy for long-term retention
  tiering_policy {
    archived_restore_point {
      mode = "TierAfter"
      duration = 180
      duration_type = "Days"
    }
  }
}

# File Share Backup Policy
resource "azurerm_backup_policy_file_share" "main" {
  for_each            = { for policy in var.backup_policies : policy.name => policy if var.enable_file_backup }
  name                = "${each.value.name}-files"
  resource_group_name = var.resource_group_name
  recovery_vault_name = azurerm_recovery_services_vault.main.name
  
  timezone = each.value.timezone
  
  backup {
    frequency = each.value.backup_frequency
    time      = each.value.backup_time
  }
  
  retention_daily {
    count = each.value.retention_daily
  }
  
  dynamic "retention_weekly" {
    for_each = each.value.retention_weekly > 0 ? [1] : []
    content {
      count    = each.value.retention_weekly
      weekdays = ["Sunday", "Wednesday", "Friday"]
    }
  }
  
  dynamic "retention_monthly" {
    for_each = each.value.retention_monthly > 0 ? [1] : []
    content {
      count    = each.value.retention_monthly
      weekdays = ["Sunday"]
      weeks    = ["First", "Last"]
    }
  }
  
  dynamic "retention_yearly" {
    for_each = each.value.retention_yearly > 0 ? [1] : []
    content {
      count    = each.value.retention_yearly
      weekdays = ["Sunday"]
      weeks    = ["Last"]
      months   = ["January", "July"]
    }
  }
}

# Backup Container for Storage Account
resource "azurerm_backup_container_storage_account" "main" {
  count               = var.enable_file_backup ? 1 : 0
  resource_group_name = var.resource_group_name
  recovery_vault_name = azurerm_recovery_services_vault.main.name
  storage_account_id  = azurerm_storage_account.backup.id
}

# Storage Account for backup staging
resource "azurerm_storage_account" "backup" {
  name                     = replace("${var.name_prefix}backup", "-", "")
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = var.storage_type == "GeoRedundant" ? "GRS" : "LRS"
  min_tls_version         = "TLS1_2"
  
  # Enable versioning and soft delete
  blob_properties {
    versioning_enabled = true
    
    delete_retention_policy {
      days = 365
    }
    
    container_delete_retention_policy {
      days = 90
    }
    
    restore_policy {
      days = 30
    }
  }
  
  # Network security
  network_rules {
    default_action = "Deny"
    bypass         = ["AzureServices", "Logging", "Metrics"]
  }
  
  # Lifecycle management
  lifecycle {
    prevent_destroy = true
  }
  
  tags = var.tags
}

# Backup Protected Items (Example for PostgreSQL)
resource "azurerm_backup_protected_vm" "example" {
  count               = 0  # Set to actual count when VMs are available
  resource_group_name = var.resource_group_name
  recovery_vault_name = azurerm_recovery_services_vault.main.name
  source_vm_id        = "vm_id_placeholder"
  backup_policy_id    = azurerm_backup_policy_vm.main[var.backup_policies[0].name].id
}

# Role Assignments for Backup
resource "azurerm_role_assignment" "vault_backup_contributor" {
  scope                = azurerm_recovery_services_vault.main.id
  role_definition_name = "Backup Contributor"
  principal_id         = azurerm_recovery_services_vault.main.identity[0].principal_id
}

resource "azurerm_role_assignment" "storage_backup_contributor" {
  scope                = azurerm_storage_account.backup.id
  role_definition_name = "Storage Account Backup Contributor"
  principal_id         = azurerm_recovery_services_vault.main.identity[0].principal_id
}

# Private Endpoint for Recovery Services Vault
resource "azurerm_private_endpoint" "vault" {
  name                = "${var.name_prefix}-rsv-pe"
  location            = var.location
  resource_group_name = var.resource_group_name
  subnet_id           = data.azurerm_subnet.private_endpoints.id
  
  private_service_connection {
    name                           = "${var.name_prefix}-rsv-psc"
    private_connection_resource_id = azurerm_recovery_services_vault.main.id
    subresource_names              = ["AzureBackup"]
    is_manual_connection           = false
  }
  
  tags = var.tags
}

# Data source for subnet (should be provided by networking module)
data "azurerm_subnet" "private_endpoints" {
  name                 = "${var.name_prefix}-pe-subnet"
  virtual_network_name = "${var.name_prefix}-vnet"
  resource_group_name  = var.resource_group_name
}

# Diagnostic Settings for Vault
resource "azurerm_monitor_diagnostic_setting" "vault" {
  name                       = "${var.name_prefix}-rsv-diagnostics"
  target_resource_id         = azurerm_recovery_services_vault.main.id
  log_analytics_workspace_id = data.azurerm_log_analytics_workspace.main.id
  
  enabled_log {
    category = "AzureBackupReport"
    
    retention_policy {
      enabled = true
      days    = 90
    }
  }
  
  enabled_log {
    category = "CoreAzureBackup"
    
    retention_policy {
      enabled = true
      days    = 90
    }
  }
  
  enabled_log {
    category = "AddonAzureBackupJobs"
    
    retention_policy {
      enabled = true
      days    = 90
    }
  }
  
  enabled_log {
    category = "AddonAzureBackupAlerts"
    
    retention_policy {
      enabled = true
      days    = 90
    }
  }
  
  enabled_log {
    category = "AddonAzureBackupPolicy"
    
    retention_policy {
      enabled = true
      days    = 90
    }
  }
  
  metric {
    category = "AllMetrics"
    
    retention_policy {
      enabled = true
      days    = 90
    }
  }
}

# Data source for Log Analytics workspace
data "azurerm_log_analytics_workspace" "main" {
  name                = "${var.name_prefix}-law"
  resource_group_name = var.resource_group_name
}

# Outputs
output "vault_id" {
  value = azurerm_recovery_services_vault.main.id
}

output "vault_name" {
  value = azurerm_recovery_services_vault.main.name
}

output "backup_storage_account_id" {
  value = azurerm_storage_account.backup.id
}

output "backup_storage_account_name" {
  value = azurerm_storage_account.backup.name
}

output "backup_storage_primary_access_key" {
  value     = azurerm_storage_account.backup.primary_access_key
  sensitive = true
}

output "backup_policies" {
  value = {
    vm_policies        = { for k, v in azurerm_backup_policy_vm.main : k => v.id }
    file_share_policies = { for k, v in azurerm_backup_policy_file_share.main : k => v.id }
  }
}

output "vault_identity_principal_id" {
  value = azurerm_recovery_services_vault.main.identity[0].principal_id
}