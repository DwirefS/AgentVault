# AgentVault™ Security Module
# Comprehensive security infrastructure with Key Vault, encryption, RBAC, and compliance

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

variable "key_vault_sku" {
  description = "SKU for Key Vault (standard or premium)"
  type        = string
  default     = "premium"
}

variable "enable_hsm_protection" {
  description = "Enable Hardware Security Module protection"
  type        = bool
  default     = false
}

variable "encryption_keys" {
  description = "List of encryption keys to create"
  type = list(object({
    name          = string
    key_type      = string
    key_size      = number
    key_opts      = list(string)
    rotation_days = number
  }))
  default = []
}

variable "admin_object_ids" {
  description = "Azure AD object IDs for administrators"
  type        = list(string)
}

variable "subnet_id" {
  description = "Subnet ID for private endpoints"
  type        = string
}

variable "allowed_ips" {
  description = "Allowed IP addresses for Key Vault access"
  type        = list(string)
  default     = []
}

variable "enable_diagnostic_logs" {
  description = "Enable diagnostic logging"
  type        = bool
  default     = true
}

variable "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID for diagnostics"
  type        = string
  default     = ""
}

variable "enable_advanced_threat_protection" {
  description = "Enable advanced threat protection"
  type        = bool
  default     = true
}

variable "enable_vulnerability_assessment" {
  description = "Enable vulnerability assessment"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Data sources
data "azurerm_client_config" "current" {}

# Random suffix for globally unique names
resource "random_string" "key_vault_suffix" {
  length  = 4
  special = false
  upper   = false
}

# Key Vault
resource "azurerm_key_vault" "main" {
  name                = "${var.name_prefix}-kv-${random_string.key_vault_suffix.result}"
  location            = var.location
  resource_group_name = var.resource_group_name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name           = var.key_vault_sku
  
  # Security features
  enabled_for_deployment          = true
  enabled_for_disk_encryption     = true
  enabled_for_template_deployment = true
  enable_rbac_authorization      = true
  purge_protection_enabled       = true
  soft_delete_retention_days     = 90
  
  # Network security
  network_acls {
    default_action             = "Deny"
    bypass                     = "AzureServices"
    ip_rules                   = var.allowed_ips
    virtual_network_subnet_ids = [var.subnet_id]
  }
  
  tags = var.tags
}

# Managed HSM (if enabled)
resource "azurerm_key_vault_managed_hardware_security_module" "main" {
  count               = var.enable_hsm_protection ? 1 : 0
  name                = "${var.name_prefix}-hsm"
  resource_group_name = var.resource_group_name
  location            = var.location
  sku_name           = "Standard_B1"
  tenant_id          = data.azurerm_client_config.current.tenant_id
  
  # Administrator configuration
  admin_object_ids = var.admin_object_ids
  
  # Network rules
  network_acls {
    default_action = "Deny"
    bypass         = "AzureServices"
  }
  
  # Security settings
  purge_protection_enabled   = true
  soft_delete_retention_days = 90
  
  tags = var.tags
}

# Encryption keys
resource "azurerm_key_vault_key" "encryption" {
  for_each = { for key in var.encryption_keys : key.name => key }
  
  name         = each.value.name
  key_vault_id = azurerm_key_vault.main.id
  key_type     = each.value.key_type
  key_size     = each.value.key_size
  key_opts     = each.value.key_opts
  
  # Rotation policy
  rotation_policy {
    automatic {
      time_before_expiry = "P30D"
    }
    
    expire_after         = "P${each.value.rotation_days}D"
    notify_before_expiry = "P29D"
  }
  
  tags = var.tags
  
  depends_on = [azurerm_role_assignment.key_vault_crypto_officer]
}

# User-assigned managed identities
resource "azurerm_user_assigned_identity" "main" {
  name                = "${var.name_prefix}-identity"
  location            = var.location
  resource_group_name = var.resource_group_name
  tags                = var.tags
}

resource "azurerm_user_assigned_identity" "aks" {
  name                = "${var.name_prefix}-aks-identity"
  location            = var.location
  resource_group_name = var.resource_group_name
  tags                = var.tags
}

resource "azurerm_user_assigned_identity" "anf" {
  name                = "${var.name_prefix}-anf-identity"
  location            = var.location
  resource_group_name = var.resource_group_name
  tags                = var.tags
}

resource "azurerm_user_assigned_identity" "backup" {
  name                = "${var.name_prefix}-backup-identity"
  location            = var.location
  resource_group_name = var.resource_group_name
  tags                = var.tags
}

# RBAC Role Assignments
# Key Vault Crypto Officer for current client
resource "azurerm_role_assignment" "key_vault_crypto_officer" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Crypto Officer"
  principal_id         = data.azurerm_client_config.current.object_id
}

# Key Vault Crypto Officer for admins
resource "azurerm_role_assignment" "key_vault_admin_crypto_officer" {
  for_each             = toset(var.admin_object_ids)
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Crypto Officer"
  principal_id         = each.value
}

# Key Vault Crypto User for managed identities
resource "azurerm_role_assignment" "key_vault_crypto_user_main" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Crypto User"
  principal_id         = azurerm_user_assigned_identity.main.principal_id
}

resource "azurerm_role_assignment" "key_vault_crypto_user_aks" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Crypto User"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

# Key Vault Secrets User for AKS
resource "azurerm_role_assignment" "key_vault_secrets_user_aks" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.aks.principal_id
}

# Storage Account for audit logs
resource "azurerm_storage_account" "audit" {
  name                     = replace("${var.name_prefix}audit", "-", "")
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "GRS"
  account_kind             = "StorageV2"
  
  # Security settings
  min_tls_version                 = "TLS1_2"
  enable_https_traffic_only       = true
  infrastructure_encryption_enabled = true
  
  # Immutable storage for compliance
  blob_properties {
    versioning_enabled = true
    
    delete_retention_policy {
      days = 365
    }
    
    container_delete_retention_policy {
      days = 365
    }
  }
  
  # Network rules
  network_rules {
    default_action             = "Deny"
    bypass                     = ["AzureServices"]
    ip_rules                   = var.allowed_ips
    virtual_network_subnet_ids = [var.subnet_id]
  }
  
  # Advanced threat protection
  advanced_threat_protection_enabled = var.enable_advanced_threat_protection
  
  tags = var.tags
}

# Storage container for audit logs
resource "azurerm_storage_container" "audit" {
  name                  = "audit-logs"
  storage_account_name  = azurerm_storage_account.audit.name
  container_access_type = "private"
}

# Diagnostic settings for Key Vault
resource "azurerm_monitor_diagnostic_setting" "key_vault" {
  count              = var.enable_diagnostic_logs && var.log_analytics_workspace_id != "" ? 1 : 0
  name               = "${var.name_prefix}-kv-diagnostics"
  target_resource_id = azurerm_key_vault.main.id
  
  log_analytics_workspace_id = var.log_analytics_workspace_id
  storage_account_id         = azurerm_storage_account.audit.id
  
  enabled_log {
    category = "AuditEvent"
    
    retention_policy {
      enabled = true
      days    = 365
    }
  }
  
  enabled_log {
    category = "AzurePolicyEvaluationDetails"
    
    retention_policy {
      enabled = true
      days    = 365
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

# Azure Policy assignments for compliance
resource "azurerm_resource_group_policy_assignment" "encryption_at_rest" {
  name                 = "${var.name_prefix}-encrypt-at-rest"
  resource_group_id    = "/subscriptions/${data.azurerm_client_config.current.subscription_id}/resourceGroups/${var.resource_group_name}"
  policy_definition_id = "/providers/Microsoft.Authorization/policyDefinitions/0961003e-5a0a-4549-abde-af6a37f2724d"
  
  parameters = jsonencode({
    effect = {
      value = "AuditIfNotExists"
    }
  })
}

resource "azurerm_resource_group_policy_assignment" "diagnostic_logs" {
  name                 = "${var.name_prefix}-diagnostic-logs"
  resource_group_id    = "/subscriptions/${data.azurerm_client_config.current.subscription_id}/resourceGroups/${var.resource_group_name}"
  policy_definition_id = "/providers/Microsoft.Authorization/policyDefinitions/f9d614c5-c173-4d56-95a7-b4437057d193"
  
  parameters = jsonencode({
    effect = {
      value = "DeployIfNotExists"
    }
    profileName = {
      value = "${var.name_prefix}-diagnostics"
    }
    logAnalytics = {
      value = var.log_analytics_workspace_id
    }
  })
}

# Azure Security Center configuration
resource "azurerm_security_center_subscription_pricing" "defender" {
  tier          = "Standard"
  resource_type = "VirtualMachines"
}

resource "azurerm_security_center_subscription_pricing" "defender_storage" {
  tier          = "Standard"
  resource_type = "StorageAccounts"
  
  extension {
    name = "OnUploadMalwareScanning"
    additional_extension_properties = {
      isMalwareScanningEnabled = "true"
    }
  }
}

resource "azurerm_security_center_subscription_pricing" "defender_keyvault" {
  tier          = "Standard"
  resource_type = "KeyVaults"
}

resource "azurerm_security_center_subscription_pricing" "defender_arm" {
  tier          = "Standard"
  resource_type = "Arm"
}

# Private endpoint for Key Vault
resource "azurerm_private_endpoint" "key_vault" {
  name                = "${var.name_prefix}-kv-pe"
  location            = var.location
  resource_group_name = var.resource_group_name
  subnet_id           = var.subnet_id
  
  private_service_connection {
    name                           = "${var.name_prefix}-kv-psc"
    private_connection_resource_id = azurerm_key_vault.main.id
    subresource_names              = ["vault"]
    is_manual_connection           = false
  }
  
  tags = var.tags
}

# Outputs
output "key_vault_id" {
  value = azurerm_key_vault.main.id
}

output "key_vault_uri" {
  value = azurerm_key_vault.main.vault_uri
}

output "key_vault_name" {
  value = azurerm_key_vault.main.name
}

output "encryption_key_id" {
  value = length(var.encryption_keys) > 0 ? azurerm_key_vault_key.encryption[var.encryption_keys[0].name].id : ""
}

output "encryption_key_ids" {
  value = { for k, v in azurerm_key_vault_key.encryption : k => v.id }
}

output "managed_identity_id" {
  value = azurerm_user_assigned_identity.main.id
}

output "managed_identity_principal_id" {
  value = azurerm_user_assigned_identity.main.principal_id
}

output "managed_identity_client_id" {
  value = azurerm_user_assigned_identity.main.client_id
}

output "aks_identity_id" {
  value = azurerm_user_assigned_identity.aks.id
}

output "aks_identity_principal_id" {
  value = azurerm_user_assigned_identity.aks.principal_id
}

output "aks_identity_client_id" {
  value = azurerm_user_assigned_identity.aks.client_id
}

output "anf_identity_id" {
  value = azurerm_user_assigned_identity.anf.id
}

output "backup_identity_id" {
  value = azurerm_user_assigned_identity.backup.id
}

output "audit_storage_account_id" {
  value = azurerm_storage_account.audit.id
}

output "audit_storage_account_name" {
  value = azurerm_storage_account.audit.name
}

output "hsm_uri" {
  value = var.enable_hsm_protection ? azurerm_key_vault_managed_hardware_security_module.main[0].hsm_uri : ""
}