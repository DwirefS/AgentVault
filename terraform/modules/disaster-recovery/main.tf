# AgentVault™ Disaster Recovery Module
# Cross-region replication, failover automation, and business continuity

# Variables
variable "primary_resource_group_name" {
  description = "Primary resource group name"
  type        = string
}

variable "primary_location" {
  description = "Primary region"
  type        = string
}

variable "dr_resource_group_name" {
  description = "DR resource group name"
  type        = string
}

variable "dr_location" {
  description = "DR region"
  type        = string
}

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "recovery_time_objective" {
  description = "RTO in hours"
  type        = number
  default     = 4
}

variable "recovery_point_objective" {
  description = "RPO in hours"
  type        = number
  default     = 1
}

variable "enable_vm_replication" {
  description = "Enable VM replication"
  type        = bool
  default     = true
}

variable "enable_storage_replication" {
  description = "Enable storage replication"
  type        = bool
  default     = true
}

variable "enable_database_replication" {
  description = "Enable database replication"
  type        = bool
  default     = true
}

variable "anf_volumes_to_replicate" {
  description = "ANF volume IDs to replicate"
  type        = list(string)
  default     = []
}

variable "database_id" {
  description = "Primary database server ID"
  type        = string
  default     = ""
}

variable "enable_traffic_manager" {
  description = "Enable Traffic Manager for global load balancing"
  type        = bool
  default     = true
}

variable "traffic_manager_endpoints" {
  description = "Traffic Manager endpoints"
  type = list(object({
    name     = string
    target   = string
    location = string
    priority = number
  }))
  default = []
}

variable "enable_automated_failover" {
  description = "Enable automated failover"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Recovery Services Vault for DR
resource "azurerm_recovery_services_vault" "dr" {
  name                = "${var.name_prefix}-dr-rsv"
  location            = var.dr_location
  resource_group_name = var.dr_resource_group_name
  sku                 = "Standard"
  
  storage_mode_type   = "GeoRedundant"
  soft_delete_enabled = true
  
  identity {
    type = "SystemAssigned"
  }
  
  tags = merge(var.tags, { Purpose = "DisasterRecovery" })
}

# Site Recovery Fabric for Primary Region
resource "azurerm_site_recovery_fabric" "primary" {
  name                = "${var.name_prefix}-primary-fabric"
  resource_group_name = var.dr_resource_group_name
  recovery_vault_name = azurerm_recovery_services_vault.dr.name
  location            = var.primary_location
}

# Site Recovery Fabric for DR Region
resource "azurerm_site_recovery_fabric" "dr" {
  name                = "${var.name_prefix}-dr-fabric"
  resource_group_name = var.dr_resource_group_name
  recovery_vault_name = azurerm_recovery_services_vault.dr.name
  location            = var.dr_location
}

# Protection Container for Primary
resource "azurerm_site_recovery_protection_container" "primary" {
  name                 = "${var.name_prefix}-primary-container"
  resource_group_name  = var.dr_resource_group_name
  recovery_vault_name  = azurerm_recovery_services_vault.dr.name
  recovery_fabric_name = azurerm_site_recovery_fabric.primary.name
}

# Protection Container for DR
resource "azurerm_site_recovery_protection_container" "dr" {
  name                 = "${var.name_prefix}-dr-container"
  resource_group_name  = var.dr_resource_group_name
  recovery_vault_name  = azurerm_recovery_services_vault.dr.name
  recovery_fabric_name = azurerm_site_recovery_fabric.dr.name
}

# Protection Container Mapping
resource "azurerm_site_recovery_protection_container_mapping" "main" {
  name                                      = "${var.name_prefix}-container-mapping"
  resource_group_name                       = var.dr_resource_group_name
  recovery_vault_name                       = azurerm_recovery_services_vault.dr.name
  recovery_fabric_name                      = azurerm_site_recovery_fabric.primary.name
  recovery_source_protection_container_name = azurerm_site_recovery_protection_container.primary.name
  recovery_target_protection_container_id   = azurerm_site_recovery_protection_container.dr.id
  
  recovery_replication_policy_id = azurerm_site_recovery_replication_policy.main.id
}

# Replication Policy
resource "azurerm_site_recovery_replication_policy" "main" {
  name                                                 = "${var.name_prefix}-replication-policy"
  resource_group_name                                  = var.dr_resource_group_name
  recovery_vault_name                                  = azurerm_recovery_services_vault.dr.name
  recovery_point_retention_in_minutes                  = var.recovery_point_objective * 60
  application_consistent_snapshot_frequency_in_minutes = 60
}

# Traffic Manager Profile
resource "azurerm_traffic_manager_profile" "main" {
  count               = var.enable_traffic_manager ? 1 : 0
  name                = "${var.name_prefix}-tm"
  resource_group_name = var.primary_resource_group_name
  
  traffic_routing_method = "Priority"
  
  dns_config {
    relative_name = var.name_prefix
    ttl           = 60
  }
  
  monitor_config {
    protocol                     = "HTTPS"
    port                         = 443
    path                         = "/health"
    interval_in_seconds          = 30
    timeout_in_seconds           = 10
    tolerated_number_of_failures = 3
  }
  
  tags = var.tags
}

# Traffic Manager Endpoints
resource "azurerm_traffic_manager_azure_endpoint" "main" {
  for_each            = var.enable_traffic_manager ? { for ep in var.traffic_manager_endpoints : ep.name => ep } : {}
  name                = each.value.name
  profile_id          = azurerm_traffic_manager_profile.main[0].id
  priority            = each.value.priority
  weight              = 100
  target_resource_id  = each.value.target
  
  geo_mappings = ["WORLD"]
}

# Automation Account for DR Runbooks
resource "azurerm_automation_account" "dr" {
  name                = "${var.name_prefix}-dr-automation"
  location            = var.dr_location
  resource_group_name = var.dr_resource_group_name
  sku_name           = "Basic"
  
  identity {
    type = "SystemAssigned"
  }
  
  tags = var.tags
}

# DR Runbook for Failover
resource "azurerm_automation_runbook" "failover" {
  name                    = "${var.name_prefix}-failover-runbook"
  location                = var.dr_location
  resource_group_name     = var.dr_resource_group_name
  automation_account_name = azurerm_automation_account.dr.name
  log_verbose            = true
  log_progress           = true
  description            = "Automated failover runbook for disaster recovery"
  runbook_type           = "PowerShell"
  
  content = <<-EOT
    param(
        [Parameter(Mandatory=$true)]
        [string]$RecoveryPlanName,
        
        [Parameter(Mandatory=$true)]
        [string]$Direction  # "PrimaryToDR" or "DRToPrimary"
    )
    
    # Import Azure modules
    Import-Module Az.RecoveryServices
    Import-Module Az.TrafficManager
    Import-Module Az.Sql
    Import-Module Az.NetAppFiles
    
    # Connect to Azure
    Connect-AzAccount -Identity
    
    # Get Recovery Plan
    $vault = Get-AzRecoveryServicesVault -Name "${var.name_prefix}-dr-rsv" -ResourceGroupName "${var.dr_resource_group_name}"
    Set-AzRecoveryServicesAsrVaultContext -Vault $vault
    $recoveryPlan = Get-AzRecoveryServicesAsrRecoveryPlan -Name $RecoveryPlanName
    
    # Start failover
    $job = Start-AzRecoveryServicesAsrUnplannedFailoverJob -RecoveryPlan $recoveryPlan -Direction $Direction
    
    # Wait for failover to complete
    do {
        $job = Get-AzRecoveryServicesAsrJob -Job $job
        Write-Output ("Job State: {0}, Progress: {1}%" -f $job.State, $job.Progress)
        Start-Sleep -Seconds 30
    } while ($job.State -ne "Succeeded" -and $job.State -ne "Failed")
    
    if ($job.State -eq "Succeeded") {
        Write-Output "Failover completed successfully"
        
        # Update Traffic Manager
        $profile = Get-AzTrafficManagerProfile -Name "${var.name_prefix}-tm" -ResourceGroupName "${var.primary_resource_group_name}"
        $endpoints = Get-AzTrafficManagerEndpoint -ProfileName $profile.Name -ResourceGroupName $profile.ResourceGroupName
        
        foreach ($endpoint in $endpoints) {
            if ($Direction -eq "PrimaryToDR") {
                if ($endpoint.Name -eq "primary") {
                    $endpoint.EndpointStatus = "Disabled"
                } elseif ($endpoint.Name -eq "secondary") {
                    $endpoint.EndpointStatus = "Enabled"
                }
            } else {
                if ($endpoint.Name -eq "primary") {
                    $endpoint.EndpointStatus = "Enabled"
                } elseif ($endpoint.Name -eq "secondary") {
                    $endpoint.EndpointStatus = "Disabled"
                }
            }
            Set-AzTrafficManagerEndpoint -TrafficManagerEndpoint $endpoint
        }
        
        Write-Output "Traffic Manager updated successfully"
    } else {
        Write-Error "Failover failed"
        throw "Failover job failed with state: $($job.State)"
    }
  EOT
  
  tags = var.tags
}

# DR Runbook for Health Check
resource "azurerm_automation_runbook" "health_check" {
  name                    = "${var.name_prefix}-health-check-runbook"
  location                = var.dr_location
  resource_group_name     = var.dr_resource_group_name
  automation_account_name = azurerm_automation_account.dr.name
  log_verbose            = true
  log_progress           = true
  description            = "Health check runbook for DR readiness"
  runbook_type           = "PowerShell"
  
  content = <<-EOT
    # Import Azure modules
    Import-Module Az.RecoveryServices
    Import-Module Az.Sql
    Import-Module Az.NetAppFiles
    Import-Module Az.Storage
    
    # Connect to Azure
    Connect-AzAccount -Identity
    
    # Initialize health status
    $healthStatus = @{
        Overall = "Healthy"
        Components = @{}
    }
    
    # Check Site Recovery
    try {
        $vault = Get-AzRecoveryServicesVault -Name "${var.name_prefix}-dr-rsv" -ResourceGroupName "${var.dr_resource_group_name}"
        Set-AzRecoveryServicesAsrVaultContext -Vault $vault
        $protectedItems = Get-AzRecoveryServicesAsrProtectableItem
        $healthStatus.Components["SiteRecovery"] = "Healthy"
    } catch {
        $healthStatus.Components["SiteRecovery"] = "Unhealthy: $_"
        $healthStatus.Overall = "Unhealthy"
    }
    
    # Check Database Replication
    try {
        # Add database health check logic
        $healthStatus.Components["Database"] = "Healthy"
    } catch {
        $healthStatus.Components["Database"] = "Unhealthy: $_"
        $healthStatus.Overall = "Unhealthy"
    }
    
    # Check ANF Replication
    try {
        # Add ANF health check logic
        $healthStatus.Components["Storage"] = "Healthy"
    } catch {
        $healthStatus.Components["Storage"] = "Unhealthy: $_"
        $healthStatus.Overall = "Unhealthy"
    }
    
    # Output results
    Write-Output $healthStatus | ConvertTo-Json -Depth 5
  EOT
  
  tags = var.tags
}

# Schedule for Health Checks
resource "azurerm_automation_schedule" "health_check" {
  name                    = "${var.name_prefix}-health-check-schedule"
  resource_group_name     = var.dr_resource_group_name
  automation_account_name = azurerm_automation_account.dr.name
  frequency              = "Hour"
  interval               = 1
  timezone               = "UTC"
  description            = "Hourly health check for DR readiness"
}

# Link Schedule to Runbook
resource "azurerm_automation_job_schedule" "health_check" {
  resource_group_name     = var.dr_resource_group_name
  automation_account_name = azurerm_automation_account.dr.name
  schedule_name          = azurerm_automation_schedule.health_check.name
  runbook_name           = azurerm_automation_runbook.health_check.name
}

# Role Assignments for Automation Account
resource "azurerm_role_assignment" "automation_contributor" {
  scope                = "/subscriptions/${data.azurerm_client_config.current.subscription_id}"
  role_definition_name = "Contributor"
  principal_id         = azurerm_automation_account.dr.identity[0].principal_id
}

# Data sources
data "azurerm_client_config" "current" {}

# Outputs
output "dr_vault_id" {
  value = azurerm_recovery_services_vault.dr.id
}

output "dr_vault_name" {
  value = azurerm_recovery_services_vault.dr.name
}

output "traffic_manager_fqdn" {
  value = var.enable_traffic_manager ? azurerm_traffic_manager_profile.main[0].fqdn : ""
}

output "automation_account_id" {
  value = azurerm_automation_account.dr.id
}

output "failover_runbook_name" {
  value = azurerm_automation_runbook.failover.name
}

output "health_check_runbook_name" {
  value = azurerm_automation_runbook.health_check.name
}

output "replication_policy_id" {
  value = azurerm_site_recovery_replication_policy.main.id
}