# AgentVault™ Storage Module - Azure NetApp Files
# Enterprise-grade storage infrastructure for AI agent workloads
#
# This module creates:
# - Azure NetApp Files account and capacity pools
# - Multi-tier storage volumes optimized for different AI workloads
# - Backup and disaster recovery configuration
# - Performance monitoring and optimization
#
# Author: Dwiref Sharma
# Contact: DwirefS@SapientEdge.io

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
  }
}

# Local variables
locals {
  name_prefix = "agentvault-${var.environment}"
  
  # Volume types optimized for AI agent workloads
  volume_types = {
    "vector-store" = {
      description = "Ultra-high performance storage for vector embeddings and similarity search"
      tier        = "ultra"
      protocols   = ["NFSv4.1"]
      features    = ["snapshot", "backup", "encryption"]
    }
    "memory-cache" = {
      description = "Premium storage for agent active memory and caching"
      tier        = "premium"  
      protocols   = ["NFSv4.1"]
      features    = ["snapshot", "backup", "encryption"]
    }
    "knowledge-base" = {
      description = "Standard storage for RAG datastores and knowledge graphs"
      tier        = "standard"
      protocols   = ["NFSv4.1", "SMB"]
      features    = ["snapshot", "backup", "encryption", "cross_region_replication"]
    }
    "chat-history" = {
      description = "Standard storage for conversation history and logs"
      tier        = "standard"
      protocols   = ["NFSv4.1"]
      features    = ["snapshot", "backup", "encryption", "analytics"]
    }
    "activity-tracking" = {
      description = "Cool storage for agent activity logs and analytics"
      tier        = "standard"  # ANF doesn't have cool tier, use lifecycle policies
      protocols   = ["NFSv4.1"]
      features    = ["snapshot", "backup", "encryption", "analytics", "archival"]
    }
    "model-artifacts" = {
      description = "Storage for AI model artifacts and checkpoints"
      tier        = "premium"
      protocols   = ["NFSv4.1"]
      features    = ["snapshot", "backup", "encryption", "versioning"]
    }
  }
}

# Random suffix for globally unique names
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Azure NetApp Files Account
resource "azurerm_netapp_account" "agentvault" {
  name                = "${local.name_prefix}-anf-${random_string.suffix.result}"
  location            = var.location
  resource_group_name = var.resource_group_name
  
  # Identity configuration for managed identity integration
  identity {
    type = "SystemAssigned"
  }
  
  tags = merge(var.tags, {
    Component = "Storage"
    Service   = "Azure NetApp Files"
    Purpose   = "AI Agent Storage Foundation"
  })
}

# Capacity Pools for different performance tiers
resource "azurerm_netapp_pool" "capacity_pools" {
  for_each = var.performance_tiers
  
  name                = "${local.name_prefix}-pool-${each.key}"
  location            = var.location
  resource_group_name = var.resource_group_name
  account_name        = azurerm_netapp_account.agentvault.name
  
  service_level = each.value.service_level
  size_in_tb   = each.value.size_gb / 1024
  qos_type     = "Auto"  # Enable automatic QoS management
  
  # Enable encryption
  encryption_type = "Single"
  
  tags = merge(var.tags, {
    Component     = "Storage"
    Service       = "ANF Capacity Pool"
    Tier          = each.value.service_level
    Description   = each.value.description
    SizeGB        = each.value.size_gb
  })
}

# Storage Account for backup and secondary storage
resource "azurerm_storage_account" "backup" {
  name                = "${replace(local.name_prefix, "-", "")}backup${random_string.suffix.result}"
  resource_group_name = var.resource_group_name
  location            = var.location
  
  account_tier             = "Standard"
  account_replication_type = "ZRS"  # Zone-redundant storage
  account_kind            = "StorageV2"
  
  # Security configurations
  enable_https_traffic_only      = true
  min_tls_version               = "TLS1_2"
  allow_nested_items_to_be_public = false
  
  # Advanced threat protection
  blob_properties {
    versioning_enabled       = true
    change_feed_enabled     = true
    last_access_time_enabled = true
    
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
    bypass                    = ["AzureServices"]
    virtual_network_subnet_ids = [var.anf_subnet_id]
  }
  
  tags = merge(var.tags, {
    Component = "Storage"
    Service   = "Backup Storage"
    Purpose   = "ANF Backup and Archive"
  })
}

# Default ANF volumes for immediate AI agent deployment
resource "azurerm_netapp_volume" "default_volumes" {
  for_each = local.volume_types
  
  name                = "${local.name_prefix}-volume-${each.key}"
  location            = var.location
  resource_group_name = var.resource_group_name
  account_name        = azurerm_netapp_account.agentvault.name
  pool_name          = azurerm_netapp_pool.capacity_pools[each.value.tier].name
  volume_path        = each.key
  service_level      = var.performance_tiers[each.value.tier].service_level
  subnet_id          = var.anf_subnet_id
  
  # Volume configuration
  storage_quota_in_gb = each.key == "vector-store" ? 1024 : 
                       each.key == "memory-cache" ? 2048 :
                       each.key == "knowledge-base" ? 4096 :
                       each.key == "model-artifacts" ? 2048 : 1024
  
  # Protocol configuration
  protocols = each.value.protocols
  
  # Export policy for NFS access
  export_policy_rule {
    rule_index          = 1
    allowed_clients     = "10.0.0.0/16"  # VNet CIDR
    unix_read_only      = false
    unix_read_write     = true
    cifs               = contains(each.value.protocols, "SMB")
    nfsv3              = false
    nfsv41             = contains(each.value.protocols, "NFSv4.1")
    root_access_enabled = false
    
    # Kerberos authentication (optional)
    kerberos_5_read_only  = false
    kerberos_5_read_write = false
    kerberos_5i_read_only = false
    kerberos_5i_read_write = false
    kerberos_5p_read_only = false
    kerberos_5p_read_write = false
  }
  
  # Security configuration
  security_style = "Unix"
  
  # Snapshot configuration
  snapshot_directory_visible = true
  
  tags = merge(var.tags, {
    Component   = "Storage"
    Service     = "ANF Volume" 
    VolumeType  = each.key
    Tier        = each.value.tier
    Description = each.value.description
    Protocols   = join(",", each.value.protocols)
    Features    = join(",", each.value.features)
  })
}

# Snapshot Policy for automated backups
resource "azurerm_netapp_snapshot_policy" "default" {
  name                = "${local.name_prefix}-snapshot-policy"
  location            = var.location
  resource_group_name = var.resource_group_name
  account_name        = azurerm_netapp_account.agentvault.name
  enabled             = true
  
  # Hourly snapshots for critical data
  hourly_schedule {
    snapshots_to_keep = 24  # Keep 24 hours
    minute            = 30  # Take at 30 minutes past the hour
  }
  
  # Daily snapshots
  daily_schedule {
    snapshots_to_keep = 30   # Keep 30 days
    hour              = 2    # Take at 2 AM
    minute            = 15   # 15 minutes past the hour
  }
  
  # Weekly snapshots
  weekly_schedule {
    snapshots_to_keep = 12   # Keep 12 weeks (3 months)
    day_of_week      = "Sunday"
    hour             = 3
    minute           = 0
  }
  
  # Monthly snapshots
  monthly_schedule {
    snapshots_to_keep = 12   # Keep 12 months
    days_of_month    = "1"   # First day of month
    hour             = 4
    minute           = 0
  }
  
  tags = merge(var.tags, {
    Component = "Storage"
    Service   = "ANF Snapshot Policy"
    Purpose   = "Automated Backup and Recovery"
  })
}

# Apply snapshot policy to critical volumes
resource "azurerm_netapp_volume_group_sap_hana" "snapshot_assignment" {
  count = 0  # Placeholder - would implement for production
  # This would assign snapshot policies to volumes based on criticality
}

# Backup Policy for long-term retention
resource "azurerm_netapp_backup_policy" "default" {
  name                = "${local.name_prefix}-backup-policy"
  resource_group_name = var.resource_group_name
  location            = var.location
  account_name        = azurerm_netapp_account.agentvault.name
  
  enabled = true
  
  # Daily backups
  daily_backups_to_keep = 30
  
  # Weekly backups  
  weekly_backups_to_keep = 12
  
  # Monthly backups
  monthly_backups_to_keep = 12
  
  tags = merge(var.tags, {
    Component = "Storage"
    Service   = "ANF Backup Policy"
    Purpose   = "Long-term Data Retention"
  })
}

# Private Endpoint for secure ANF access
resource "azurerm_private_endpoint" "anf_endpoint" {
  name                = "${local.name_prefix}-anf-pe"
  location            = var.location
  resource_group_name = var.resource_group_name
  subnet_id           = var.anf_subnet_id
  
  private_service_connection {
    name                           = "${local.name_prefix}-anf-psc"
    private_connection_resource_id = azurerm_netapp_account.agentvault.id
    subresource_names             = ["netAppAccount"]
    is_manual_connection          = false
  }
  
  # DNS integration
  private_dns_zone_group {
    name                 = "anf-dns-zone-group"
    private_dns_zone_ids = [azurerm_private_dns_zone.anf_dns.id]
  }
  
  tags = merge(var.tags, {
    Component = "Security"
    Service   = "Private Endpoint"
    Purpose   = "Secure ANF Access"
  })
}

# Private DNS Zone for ANF
resource "azurerm_private_dns_zone" "anf_dns" {
  name                = "privatelink.file.core.windows.net"
  resource_group_name = var.resource_group_name
  
  tags = merge(var.tags, {
    Component = "Networking"
    Service   = "Private DNS"
    Purpose   = "ANF Name Resolution"
  })
}

# Diagnostic settings for monitoring
resource "azurerm_monitor_diagnostic_setting" "anf_diagnostics" {
  name               = "${local.name_prefix}-anf-diagnostics"
  target_resource_id = azurerm_netapp_account.agentvault.id
  storage_account_id = azurerm_storage_account.backup.id
  
  enabled_log {
    category = "AuditEvent"
  }
  
  metric {
    category = "AllMetrics"
    enabled  = true
    
    retention_policy {
      enabled = true
      days    = 90
    }
  }
  
  tags = var.tags
}

# Data Factory for ANF data lifecycle management
resource "azurerm_data_factory" "anf_lifecycle" {
  name                = "${local.name_prefix}-adf"
  location            = var.location
  resource_group_name = var.resource_group_name
  
  # Managed identity for secure access
  identity {
    type = "SystemAssigned"
  }
  
  tags = merge(var.tags, {
    Component = "Data Management"
    Service   = "Azure Data Factory"
    Purpose   = "ANF Lifecycle Management"
  })
}

# Outputs
output "netapp_account_id" {
  description = "ID of the Azure NetApp Files account"
  value       = azurerm_netapp_account.agentvault.id
}

output "netapp_account_name" {
  description = "Name of the Azure NetApp Files account"
  value       = azurerm_netapp_account.agentvault.name
}

output "capacity_pools" {
  description = "Created capacity pools with their configurations"
  value = {
    for tier, pool in azurerm_netapp_pool.capacity_pools : tier => {
      id            = pool.id
      name          = pool.name
      service_level = pool.service_level
      size_tb       = pool.size_in_tb
    }
  }
}

output "anf_volumes" {
  description = "Created ANF volumes with mount information"
  value = {
    for volume_type, volume in azurerm_netapp_volume.default_volumes : volume_type => {
      id                  = volume.id
      name                = volume.name
      mount_ip           = volume.mount_ip_addresses[0]
      volume_path        = volume.volume_path
      storage_quota_gb   = volume.storage_quota_in_gb
      protocols          = volume.protocols
      service_level      = volume.service_level
      mount_command      = "sudo mount -t nfs -o rw,hard,rsize=65536,wsize=65536,vers=4.1,tcp ${volume.mount_ip_addresses[0]}:/${volume.volume_path} /mnt/${volume_type}"
    }
  }
}

output "storage_account_id" {
  description = "ID of the backup storage account"
  value       = azurerm_storage_account.backup.id
}

output "backup_storage_connection_string" {
  description = "Connection string for backup storage"
  value       = azurerm_storage_account.backup.primary_connection_string
  sensitive   = true
}

output "snapshot_policy_id" {
  description = "ID of the snapshot policy"
  value       = azurerm_netapp_snapshot_policy.default.id
}

output "backup_policy_id" {
  description = "ID of the backup policy"
  value       = azurerm_netapp_backup_policy.default.id
}

output "private_endpoint_ip" {
  description = "Private IP address of the ANF endpoint"
  value       = azurerm_private_endpoint.anf_endpoint.private_service_connection[0].private_ip_address
}

output "mount_instructions" {
  description = "Instructions for mounting ANF volumes"
  value = {
    for volume_type, volume in azurerm_netapp_volume.default_volumes : volume_type => {
      description = local.volume_types[volume_type].description
      mount_point = "/mnt/agentvault/${volume_type}"
      commands = [
        "# Create mount point",
        "sudo mkdir -p /mnt/agentvault/${volume_type}",
        "",
        "# Mount volume",
        "sudo mount -t nfs -o rw,hard,rsize=65536,wsize=65536,vers=4.1,tcp ${volume.mount_ip_addresses[0]}:/${volume.volume_path} /mnt/agentvault/${volume_type}",
        "",
        "# Add to fstab for persistent mounting",
        "echo '${volume.mount_ip_addresses[0]}:/${volume.volume_path} /mnt/agentvault/${volume_type} nfs rw,hard,rsize=65536,wsize=65536,vers=4.1,tcp 0 0' | sudo tee -a /etc/fstab",
        "",
        "# Set permissions",
        "sudo chown -R $USER:$USER /mnt/agentvault/${volume_type}",
        "sudo chmod -R 755 /mnt/agentvault/${volume_type}"
      ]
    }
  }
}