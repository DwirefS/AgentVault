# AgentVault™ - Enterprise AI Agent Storage Platform
# Complete Azure Infrastructure with HA, DR, Security & Compliance
# 
# This Terraform configuration deploys a production-ready AgentVault™
# infrastructure with full enterprise features including:
# - High Availability across availability zones
# - Disaster Recovery with cross-region replication
# - End-to-end encryption and security
# - Comprehensive monitoring and alerting
# - Automated backup and recovery
# - Cost optimization and governance
#
# Author: Dwiref Sharma
# Contact: DwirefS@SapientEdge.io

terraform {
  required_version = ">= 1.3.0"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.85.0"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 2.47.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.24.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0.0"
    }
  }
  
  # Backend configuration for state management with locking
  backend "azurerm" {
    resource_group_name  = "agentvault-terraform-state"
    storage_account_name = "agentvaultterraform"
    container_name       = "tfstate"
    key                  = "agentvault.terraform.tfstate"
    
    # Enable state locking
    use_azuread_auth = true
  }
}

# Configure providers
provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = false
      recover_soft_deleted_key_vaults = true
    }
    resource_group {
      prevent_deletion_if_contains_resources = true
    }
  }
}

provider "kubernetes" {
  host                   = module.aks.kube_config.0.host
  client_certificate     = base64decode(module.aks.kube_config.0.client_certificate)
  client_key             = base64decode(module.aks.kube_config.0.client_key)
  cluster_ca_certificate = base64decode(module.aks.kube_config.0.cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = module.aks.kube_config.0.host
    client_certificate     = base64decode(module.aks.kube_config.0.client_certificate)
    client_key             = base64decode(module.aks.kube_config.0.client_key)
    cluster_ca_certificate = base64decode(module.aks.kube_config.0.cluster_ca_certificate)
  }
}

# Data sources
data "azurerm_client_config" "current" {}

data "azuread_client_config" "current" {}

# Local variables
locals {
  common_tags = merge(
    var.common_tags,
    {
      Environment         = var.environment
      Project            = "AgentVault"
      ManagedBy          = "Terraform"
      CreatedBy          = data.azuread_client_config.current.object_id
      CreatedDate        = timestamp()
      LastModified       = timestamp()
      CostCenter         = var.cost_center
      BusinessUnit       = var.business_unit
      DataClassification = "Confidential"
      Compliance         = "SOC2,ISO27001,HIPAA"
      BackupRequired     = "true"
      DREnabled          = var.enable_disaster_recovery
    }
  )
  
  # Region mapping for DR
  dr_region_map = {
    "eastus"         = "westus"
    "westus"         = "eastus"
    "northeurope"    = "westeurope"
    "westeurope"     = "northeurope"
    "southeastasia"  = "eastasia"
    "eastasia"       = "southeastasia"
  }
  
  dr_location = var.enable_disaster_recovery ? local.dr_region_map[var.location] : ""
  
  # Naming convention
  name_prefix = "${var.project_name}-${var.environment}"
  
  # Security settings
  encryption_key_name = "${local.name_prefix}-encryption-key"
  
  # Network settings
  vnet_address_space = var.vnet_address_space
  subnet_count       = 6 # AKS, ANF, Private Endpoints, Application Gateway, Bastion, Database
}

# Resource Groups
resource "azurerm_resource_group" "main" {
  name     = "${local.name_prefix}-rg"
  location = var.location
  tags     = local.common_tags
}

resource "azurerm_resource_group" "dr" {
  count    = var.enable_disaster_recovery ? 1 : 0
  name     = "${local.name_prefix}-dr-rg"
  location = local.dr_location
  tags     = merge(local.common_tags, { Purpose = "DisasterRecovery" })
}

resource "azurerm_resource_group" "backup" {
  name     = "${local.name_prefix}-backup-rg"
  location = var.location
  tags     = merge(local.common_tags, { Purpose = "Backup" })
}

# Resource Lock for production
resource "azurerm_management_lock" "main" {
  count      = var.environment == "prod" ? 1 : 0
  name       = "${local.name_prefix}-lock"
  scope      = azurerm_resource_group.main.id
  lock_level = "CanNotDelete"
  notes      = "Production resource group - deletion protection enabled"
}

# Networking Module
module "networking" {
  source = "./modules/networking"
  
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name_prefix         = local.name_prefix
  
  vnet_address_space  = local.vnet_address_space
  enable_ddos_protection = var.environment == "prod"
  enable_firewall     = var.enable_azure_firewall
  enable_bastion      = var.enable_bastion
  
  # Network security
  enable_network_watcher = true
  enable_flow_logs      = true
  
  # Private DNS zones
  create_private_dns_zones = true
  private_dns_zones = [
    "privatelink.database.windows.net",
    "privatelink.blob.core.windows.net",
    "privatelink.file.core.windows.net",
    "privatelink.vaultcore.azure.net",
    "privatelink.azurecr.io",
    "privatelink.servicebus.windows.net",
    "privatelink.redis.cache.windows.net"
  ]
  
  tags = local.common_tags
}

# Security Module
module "security" {
  source = "./modules/security"
  
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name_prefix         = local.name_prefix
  
  # Key Vault configuration
  key_vault_sku = var.environment == "prod" ? "premium" : "standard"
  enable_hsm_protection = var.enable_hsm
  
  # Encryption keys
  encryption_keys = [
    {
      name         = local.encryption_key_name
      key_type     = "RSA-HSM"
      key_size     = 4096
      key_opts     = ["decrypt", "encrypt", "sign", "unwrapKey", "verify", "wrapKey"]
      rotation_days = 90
    },
    {
      name         = "${local.name_prefix}-backup-key"
      key_type     = "RSA-HSM"
      key_size     = 4096
      key_opts     = ["decrypt", "encrypt"]
      rotation_days = 180
    }
  ]
  
  # Access policies
  admin_object_ids = var.admin_object_ids
  
  # Network rules
  subnet_id = module.networking.private_endpoint_subnet_id
  allowed_ips = var.allowed_ips
  
  # Audit logging
  enable_diagnostic_logs = true
  log_analytics_workspace_id = module.monitoring.log_analytics_workspace_id
  
  # Advanced security
  enable_advanced_threat_protection = var.environment == "prod"
  enable_vulnerability_assessment   = true
  
  tags = local.common_tags
}

# Identity Module
module "identity" {
  source = "./modules/identity"
  
  name_prefix = local.name_prefix
  
  # Managed identities
  create_user_assigned_identity = true
  identity_name = "${local.name_prefix}-identity"
  
  # Service principals
  create_service_principals = true
  service_principals = [
    {
      name = "${local.name_prefix}-aks-sp"
      roles = ["Contributor"]
    },
    {
      name = "${local.name_prefix}-anf-sp"
      roles = ["Storage Account Contributor", "NetApp Files Contributor"]
    }
  ]
  
  # Azure AD groups
  create_security_groups = true
  security_groups = [
    {
      name        = "${local.name_prefix}-admins"
      description = "AgentVault Administrators"
      owners      = var.admin_object_ids
    },
    {
      name        = "${local.name_prefix}-developers"
      description = "AgentVault Developers"
    },
    {
      name        = "${local.name_prefix}-operators"
      description = "AgentVault Operators"
    }
  ]
  
  # App registrations
  create_app_registrations = true
  app_registrations = [
    {
      name = "${local.name_prefix}-api"
      api_permissions = [
        "User.Read",
        "Directory.Read.All"
      ]
      redirect_uris = ["https://${var.domain_name}/auth/callback"]
    }
  ]
  
  tags = local.common_tags
}

# AKS Cluster Module
module "aks" {
  source = "./modules/aks"
  
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name_prefix         = local.name_prefix
  
  # Cluster configuration
  kubernetes_version = var.kubernetes_version
  sku_tier          = var.environment == "prod" ? "Standard" : "Free"
  
  # Node pools
  system_node_pool = {
    name                = "system"
    node_count          = var.environment == "prod" ? 3 : 1
    min_count           = var.environment == "prod" ? 3 : 1
    max_count           = var.environment == "prod" ? 5 : 3
    vm_size             = var.aks_system_node_size
    availability_zones  = var.environment == "prod" ? ["1", "2", "3"] : ["1"]
    max_pods            = 30
    os_disk_size_gb     = 128
    os_disk_type        = "Managed"
    ultra_ssd_enabled   = false
  }
  
  user_node_pools = [
    {
      name                = "agents"
      node_count          = var.aks_agent_node_count
      min_count           = var.aks_agent_node_count
      max_count           = var.aks_agent_max_count
      vm_size             = var.aks_agent_node_size
      availability_zones  = var.environment == "prod" ? ["1", "2", "3"] : ["1"]
      max_pods            = 50
      os_disk_size_gb     = 256
      os_disk_type        = "Managed"
      ultra_ssd_enabled   = true
      node_labels = {
        "workload" = "agents"
        "gpu"      = "enabled"
      }
      node_taints = [
        "agents=true:NoSchedule"
      ]
    },
    {
      name                = "monitoring"
      node_count          = var.environment == "prod" ? 2 : 1
      min_count           = var.environment == "prod" ? 2 : 1
      max_count           = 3
      vm_size             = var.aks_monitoring_node_size
      availability_zones  = var.environment == "prod" ? ["1", "2"] : ["1"]
      max_pods            = 30
      os_disk_size_gb     = 128
      os_disk_type        = "Managed"
      node_labels = {
        "workload" = "monitoring"
      }
    }
  ]
  
  # Networking
  vnet_subnet_id = module.networking.aks_subnet_id
  network_plugin = "azure"
  network_policy = "calico"
  service_cidr   = "10.100.0.0/16"
  dns_service_ip = "10.100.0.10"
  
  # Security
  enable_pod_security_policy = true
  enable_azure_policy       = true
  enable_workload_identity  = true
  
  # Monitoring
  enable_monitoring = true
  log_analytics_workspace_id = module.monitoring.log_analytics_workspace_id
  
  # Addons
  enable_ingress_application_gateway = true
  enable_key_vault_secrets_provider  = true
  key_vault_id = module.security.key_vault_id
  
  # RBAC
  rbac_azure_ad_managed = true
  rbac_azure_ad_admin_group_ids = [module.identity.admin_group_id]
  
  # Backup
  enable_backup = true
  
  identity_ids = [module.identity.aks_identity_id]
  
  tags = local.common_tags
}

# Storage Module (ANF)
module "storage" {
  source = "./modules/storage"
  
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name_prefix         = local.name_prefix
  
  # NetApp account
  netapp_account_name = "${local.name_prefix}-anf"
  
  # Capacity pools
  capacity_pools = [
    {
      name          = "premium"
      size_in_tb    = var.anf_premium_pool_size
      service_level = "Premium"
      qos_type      = "Manual"
    },
    {
      name          = "standard"
      size_in_tb    = var.anf_standard_pool_size
      service_level = "Standard"
      qos_type      = "Auto"
    },
    {
      name          = "ultra"
      size_in_tb    = var.anf_ultra_pool_size
      service_level = "Ultra"
      qos_type      = "Manual"
    }
  ]
  
  # Volumes
  volumes = [
    {
      name                = "agents"
      pool_name           = "premium"
      size_in_tb          = var.anf_agents_volume_size
      throughput_in_mibps = 128
      protocols           = ["NFSv4.1"]
      snapshot_policy     = "daily"
      backup_enabled      = true
      replication_enabled = var.enable_disaster_recovery
    },
    {
      name                = "models"
      pool_name           = "standard"
      size_in_tb          = var.anf_models_volume_size
      throughput_in_mibps = 64
      protocols           = ["NFSv4.1"]
      snapshot_policy     = "weekly"
      backup_enabled      = true
      replication_enabled = var.enable_disaster_recovery
    },
    {
      name                = "artifacts"
      pool_name           = "standard"
      size_in_tb          = var.anf_artifacts_volume_size
      throughput_in_mibps = 64
      protocols           = ["NFSv4.1"]
      snapshot_policy     = "daily"
      backup_enabled      = true
      replication_enabled = var.enable_disaster_recovery
    }
  ]
  
  # Network
  subnet_id = module.networking.anf_subnet_id
  
  # Azure AD integration
  active_directory = {
    dns_servers  = var.ad_dns_servers
    domain       = var.ad_domain
    smb_server_name = "${local.name_prefix}-smb"
    username     = var.ad_username
    password     = var.ad_password
  }
  
  # Encryption
  encryption_key_source = "Microsoft.KeyVault"
  key_vault_key_id      = module.security.encryption_key_id
  
  # DR configuration
  enable_cross_region_replication = var.enable_disaster_recovery
  dr_location = local.dr_location
  
  tags = local.common_tags
}

# Database Module
module "database" {
  source = "./modules/database"
  
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name_prefix         = local.name_prefix
  
  # PostgreSQL Flexible Server
  postgresql_version = "15"
  sku_name          = var.environment == "prod" ? "GP_Standard_D4s_v3" : "B_Standard_B2s"
  
  # High Availability
  enable_high_availability = var.environment == "prod"
  high_availability_mode   = "ZoneRedundant"
  availability_zone        = "1"
  standby_availability_zone = "2"
  
  # Storage
  storage_mb = var.database_storage_mb
  backup_retention_days = var.environment == "prod" ? 35 : 7
  geo_redundant_backup_enabled = var.enable_disaster_recovery
  
  # Network
  delegated_subnet_id = module.networking.database_subnet_id
  private_dns_zone_id = module.networking.private_dns_zone_ids["database"]
  
  # Security
  ssl_enforcement_enabled = true
  ssl_minimal_tls_version = "TLS1_2"
  
  # Monitoring
  enable_threat_detection = true
  enable_audit_logs      = true
  log_analytics_workspace_id = module.monitoring.log_analytics_workspace_id
  
  # Databases
  databases = [
    {
      name      = "agentvault"
      charset   = "UTF8"
      collation = "en_US.utf8"
    }
  ]
  
  # Firewall rules (for private endpoint, this should be empty)
  firewall_rules = []
  
  # Administrator
  administrator_login    = var.database_admin_username
  administrator_password = random_password.database_admin.result
  
  tags = local.common_tags
}

# Redis Cache Module - Distributed Cache Infrastructure
module "redis" {
  source = "./modules/redis"
  
  # Basic configuration
  environment         = var.environment
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.common_tags
  
  # Networking configuration
  virtual_network_id         = module.networking.vnet_id
  cache_subnet_id           = module.networking.redis_subnet_id
  private_endpoint_subnet_id = module.networking.private_endpoint_subnet_id
  aks_subnet_start_ip       = cidrhost(module.networking.aks_subnet_cidr, 1)
  aks_subnet_end_ip         = cidrhost(module.networking.aks_subnet_cidr, -2)
  
  # Security configuration
  key_vault_id            = module.security.key_vault_id
  enable_private_endpoint = var.environment == "prod"
  
  # High availability configuration
  enable_zone_redundancy   = var.environment == "prod"
  enable_redis_enterprise  = var.environment == "prod" && var.enable_ultra_performance
  
  # Backup configuration
  backup_storage_connection_string = module.storage.backup_storage_connection_string
  
  # Monitoring configuration
  log_analytics_workspace_id = module.monitoring.log_analytics_workspace_id
  action_group_id           = module.monitoring.action_group_ids["critical"]
  
  # Performance configuration
  redis_cache_sizes = var.environment == "prod" ? {
    vectors = { capacity = 5, shard_count = 4 }  # 26GB P5 with 4 shards
    memory  = { capacity = 3, shard_count = 2 }  # 13GB P3 with 2 shards
    api     = { capacity = 2, shard_count = 1 }  # 6GB P2
    session = { capacity = 1, shard_count = 1 }  # 6GB P1
  } : {}
  
  # Alert thresholds
  alert_thresholds = {
    memory_usage_percentage = 85
    cpu_usage_percentage    = 75
    connection_count        = 9000
    cache_miss_rate        = 30
  }
  
  # Feature flags
  enable_redis_modules    = true
  enable_geo_replication  = var.enable_disaster_recovery
  geo_replication_regions = var.enable_disaster_recovery ? [azurerm_resource_group.dr[0].location] : []
}

# Monitoring Module
module "monitoring" {
  source = "./modules/monitoring"
  
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  name_prefix         = local.name_prefix
  
  # Log Analytics
  log_analytics_retention_days = var.environment == "prod" ? 90 : 30
  
  # Application Insights
  enable_application_insights = true
  application_type           = "web"
  
  # Azure Monitor
  enable_azure_monitor = true
  
  # Metrics and Alerts
  create_action_groups = true
  action_groups = [
    {
      name = "${local.name_prefix}-critical"
      email_receivers = var.alert_email_addresses
      sms_receivers   = var.environment == "prod" ? var.alert_sms_numbers : []
      webhook_receivers = var.alert_webhooks
    },
    {
      name = "${local.name_prefix}-warning"
      email_receivers = var.alert_email_addresses
    }
  ]
  
  # Prometheus and Grafana
  enable_prometheus = true
  enable_grafana   = true
  grafana_admin_user = "admin"
  
  # Dashboards
  create_dashboards = true
  dashboard_templates = [
    "aks-cluster",
    "anf-storage",
    "application-performance",
    "agent-metrics",
    "security-overview"
  ]
  
  # Workbooks
  create_workbooks = true
  
  # Data Collection Rules
  enable_container_insights = true
  enable_vm_insights       = true
  
  # Diagnostic settings for all resources
  diagnostic_settings_enabled = true
  
  tags = local.common_tags
}

# Backup Module
module "backup" {
  source = "./modules/backup"
  
  resource_group_name = azurerm_resource_group.backup.name
  location            = azurerm_resource_group.backup.location
  name_prefix         = local.name_prefix
  
  # Recovery Services Vault
  vault_sku = "Standard"
  
  # Backup policies
  backup_policies = [
    {
      name                = "daily"
      timezone            = "UTC"
      backup_frequency    = "Daily"
      backup_time         = "23:00"
      retention_daily     = 7
      retention_weekly    = 4
      retention_monthly   = 12
      retention_yearly    = 5
    },
    {
      name                = "weekly"
      timezone            = "UTC"
      backup_frequency    = "Weekly"
      backup_time         = "23:00"
      backup_weekdays     = ["Sunday"]
      retention_daily     = 0
      retention_weekly    = 8
      retention_monthly   = 12
      retention_yearly    = 5
    }
  ]
  
  # Protected items
  enable_vm_backup       = true
  enable_file_backup     = true
  enable_database_backup = true
  
  # Geo-redundancy
  storage_type = var.enable_disaster_recovery ? "GeoRedundant" : "LocallyRedundant"
  
  # Soft delete
  soft_delete_enabled = true
  
  tags = local.common_tags
}

# Disaster Recovery Module
module "disaster_recovery" {
  count = var.enable_disaster_recovery ? 1 : 0
  
  source = "./modules/disaster-recovery"
  
  primary_resource_group_name = azurerm_resource_group.main.name
  primary_location            = azurerm_resource_group.main.location
  
  dr_resource_group_name = azurerm_resource_group.dr[0].name
  dr_location            = azurerm_resource_group.dr[0].location
  
  name_prefix = local.name_prefix
  
  # Recovery plan
  recovery_time_objective = 4  # Hours
  recovery_point_objective = 1 # Hours
  
  # Replication settings
  enable_vm_replication       = true
  enable_storage_replication  = true
  enable_database_replication = true
  
  # ANF cross-region replication
  anf_volumes_to_replicate = module.storage.volume_ids
  
  # Database geo-replication
  database_id = module.database.server_id
  
  # Traffic Manager for failover
  enable_traffic_manager = true
  traffic_manager_endpoints = [
    {
      name     = "primary"
      target   = module.aks.ingress_ip
      location = var.location
      priority = 1
    },
    {
      name     = "secondary"
      target   = "" # Will be populated after DR AKS deployment
      location = local.dr_location
      priority = 2
    }
  ]
  
  # Automation
  enable_automated_failover = false # Manual failover for safety
  
  tags = merge(local.common_tags, { Purpose = "DisasterRecovery" })
}

# Cost Management Module - Cost Optimization & Budget Control
module "cost_management" {
  source = "./modules/cost-management"
  
  # Basic configuration
  environment         = var.environment
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subscription_id     = data.azurerm_client_config.current.subscription_id
  tenant_id          = data.azurerm_client_config.current.tenant_id
  tags               = local.common_tags
  
  # Budget configuration
  monthly_budget_amount = var.monthly_budget
  component_budgets = {
    networking = {
      resource_group_id = azurerm_resource_group.main.id
      monthly_amount    = var.monthly_budget * 0.20  # 20% for networking
      additional_emails = []
    }
    storage = {
      resource_group_id = azurerm_resource_group.main.id
      monthly_amount    = var.monthly_budget * 0.30  # 30% for storage
      additional_emails = []
    }
    compute = {
      resource_group_id = azurerm_resource_group.main.id
      monthly_amount    = var.monthly_budget * 0.40  # 40% for compute
      additional_emails = []
    }
  }
  
  # Alert configuration
  cost_alert_emails = var.budget_alert_emails
  cost_webhook_urls = var.cost_webhook_urls
  critical_alert_sms = var.environment == "prod" ? var.critical_alert_sms : {}
  
  # Cost export configuration
  cost_export_storage_account_id = module.monitoring.storage_account_id
  
  # Automation configuration
  timezone                    = var.timezone
  dev_shutdown_time          = "19:00"
  dev_startup_time           = "07:00"
  auto_shutdown_dev_resources = var.environment != "prod"
  auto_scale_weekends        = true
  
  # Feature flags
  enable_anomaly_detection        = true
  enable_advisor_recommendations  = true
  enable_cost_allocation         = true
  enforce_tagging_policy         = var.environment == "prod"
  
  # Optimization settings
  unused_resource_threshold_days        = 7
  low_utilization_threshold_percent     = 10
  enable_ri_recommendations             = true
  ri_recommendation_lookback_days       = 30
  ri_coverage_target_percent           = 80
  
  # Anomaly settings
  anomaly_detection_sensitivity = "Medium"
  anomaly_threshold_percent    = 20
  
  # Cost allocation
  cost_allocation_tags = ["Department", "Project", "Owner", "CostCenter", "Environment"]
  shared_resource_allocation_method = "Proportional"
  
  # Reporting
  cost_report_recipients = concat(var.budget_alert_emails, var.cost_report_additional_emails)
  report_frequency      = "Weekly"
  include_recommendations_in_report = true
}

# Random resources
resource "random_password" "database_admin" {
  length  = 32
  special = true
  upper   = true
  lower   = true
  numeric = true
}

resource "random_password" "redis_primary_key" {
  length  = 32
  special = true
}

# Outputs
output "resource_group_name" {
  value       = azurerm_resource_group.main.name
  description = "The name of the main resource group"
}

output "aks_cluster_name" {
  value       = module.aks.cluster_name
  description = "The name of the AKS cluster"
}

output "anf_account_name" {
  value       = module.storage.netapp_account_name
  description = "The name of the NetApp account"
}

output "key_vault_uri" {
  value       = module.security.key_vault_uri
  description = "The URI of the Key Vault"
  sensitive   = true
}

output "database_fqdn" {
  value       = module.database.fqdn
  description = "The FQDN of the PostgreSQL server"
  sensitive   = true
}

output "redis_connection_info" {
  value = {
    instances     = module.redis.redis_instances
    auth         = module.redis.redis_auth
    monitoring   = module.redis.monitoring_resources
    config       = module.redis.application_config
  }
  description = "Redis cache connection information and configuration"
  sensitive   = true
}

output "log_analytics_workspace_id" {
  value       = module.monitoring.log_analytics_workspace_id
  description = "The ID of the Log Analytics workspace"
}

output "application_insights_instrumentation_key" {
  value       = module.monitoring.application_insights_instrumentation_key
  description = "The instrumentation key for Application Insights"
  sensitive   = true
}

output "backup_vault_id" {
  value       = module.backup.vault_id
  description = "The ID of the Recovery Services vault"
}

output "traffic_manager_fqdn" {
  value       = var.enable_disaster_recovery ? module.disaster_recovery[0].traffic_manager_fqdn : ""
  description = "The FQDN of the Traffic Manager profile for DR"
}

output "grafana_endpoint" {
  value       = module.monitoring.grafana_endpoint
  description = "The endpoint for Grafana dashboard"
}

output "prometheus_endpoint" {
  value       = module.monitoring.prometheus_endpoint
  description = "The endpoint for Prometheus metrics"
}

output "cost_management_info" {
  value = {
    budgets               = module.cost_management.budgets
    alerts                = module.cost_management.alert_configuration
    automation            = module.cost_management.automation
    dashboards            = module.cost_management.dashboards
    optimization_insights = module.cost_management.optimization_insights
    estimated_savings_usd = module.cost_management.optimization_insights.estimated_savings_usd.total_potential
  }
  description = "Cost management configuration and insights"
}

output "cost_optimization_summary" {
  value = {
    monthly_budget_usd     = var.monthly_budget
    optimization_score     = module.cost_management.cost_management_summary.optimization_score
    potential_savings_usd  = module.cost_management.optimization_insights.estimated_savings_usd.total_potential
    immediate_actions      = module.cost_management.optimization_insights.immediate_actions
  }
  description = "Cost optimization summary and recommendations"
}