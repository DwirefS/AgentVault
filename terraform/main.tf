# AgentVault™ - Enterprise AI Agent Storage Platform
# Complete Azure NetApp Files Infrastructure as Code
# 
# This Terraform configuration deploys a production-ready AgentVault™
# infrastructure optimized for enterprise AI agent storage workloads
#
# Author: Dwiref Sharma
# Contact: DwirefS@SapientEdge.io

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 2.45"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  backend "azurerm" {
    # Configure remote state storage
    # Values will be provided via backend config file or CLI
  }
}

# Configure Azure Provider
provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
    
    cognitive_account {
      purge_soft_delete_on_destroy = true
    }
    
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# Data sources
data "azurerm_client_config" "current" {}

data "azurerm_subscription" "current" {}

# Local variables
locals {
  # Project metadata
  project_name = "agentvault"
  environment  = var.environment
  location     = var.location
  
  # Resource naming convention
  name_prefix = "${local.project_name}-${local.environment}"
  
  # Common tags applied to all resources
  common_tags = merge(var.tags, {
    Project     = "AgentVault™"
    Environment = local.environment
    ManagedBy   = "Terraform"
    Owner       = "Dwiref Sharma"
    Contact     = "DwirefS@SapientEdge.io"
    Purpose     = "Enterprise AI Agent Storage Platform"
    CreatedAt   = timestamp()
  })
  
  # Performance tiers configuration
  performance_tiers = {
    ultra = {
      service_level = "Ultra"
      size_gb      = 4096  # 4TB minimum for Ultra
      description  = "Sub-millisecond latency for vectors and active memory"
    }
    premium = {
      service_level = "Premium" 
      size_gb      = 8192  # 8TB for premium workloads
      description  = "Low latency for long-term memory and knowledge graphs"
    }
    standard = {
      service_level = "Standard"
      size_gb      = 16384  # 16TB for standard workloads
      description  = "Standard performance for chat history and warm data"
    }
  }
  
  # Network configuration
  vnet_address_space     = ["10.0.0.0/16"]
  anf_subnet_address     = ["10.0.1.0/24"]
  compute_subnet_address = ["10.0.2.0/24"]
  bastion_subnet_address = ["10.0.3.0/24"]
  
  # Security configuration
  allowed_ip_ranges = var.allowed_ip_ranges
}

# Resource Group
resource "azurerm_resource_group" "agentvault" {
  name     = "${local.name_prefix}-rg"
  location = local.location
  tags     = local.common_tags
}

# Networking Module
module "networking" {
  source = "./modules/networking"
  
  resource_group_name = azurerm_resource_group.agentvault.name
  location           = local.location
  environment        = local.environment
  
  vnet_address_space     = local.vnet_address_space
  anf_subnet_address     = local.anf_subnet_address  
  compute_subnet_address = local.compute_subnet_address
  bastion_subnet_address = local.bastion_subnet_address
  
  allowed_ip_ranges = local.allowed_ip_ranges
  
  tags = local.common_tags
}

# Security Module - Key Vault, Managed Identity, RBAC
module "security" {
  source = "./modules/security"
  
  resource_group_name = azurerm_resource_group.agentvault.name
  location           = local.location
  environment        = local.environment
  
  tenant_id           = data.azurerm_client_config.current.tenant_id
  object_id           = data.azurerm_client_config.current.object_id
  subscription_id     = data.azurerm_subscription.current.subscription_id
  
  allowed_ip_ranges = local.allowed_ip_ranges
  
  tags = local.common_tags
  
  depends_on = [module.networking]
}

# Azure NetApp Files Storage Module
module "storage" {
  source = "./modules/storage"
  
  resource_group_name = azurerm_resource_group.agentvault.name
  location           = local.location
  environment        = local.environment
  
  # Network configuration
  anf_subnet_id = module.networking.anf_subnet_id
  
  # Performance tiers
  performance_tiers = local.performance_tiers
  
  # Security configuration
  key_vault_id = module.security.key_vault_id
  managed_identity_id = module.security.storage_managed_identity_id
  
  tags = local.common_tags
  
  depends_on = [module.networking, module.security]
}

# Monitoring Module - Application Insights, Log Analytics
module "monitoring" {
  source = "./modules/monitoring"
  
  resource_group_name = azurerm_resource_group.agentvault.name
  location           = local.location
  environment        = local.environment
  
  # Storage integration
  storage_account_id = module.storage.storage_account_id
  netapp_account_id  = module.storage.netapp_account_id
  
  # Security integration  
  key_vault_id = module.security.key_vault_id
  
  tags = local.common_tags
  
  depends_on = [module.storage, module.security]
}

# Compute Module - Container Apps, AKS, Functions
module "compute" {
  source = "./modules/compute"
  
  resource_group_name = azurerm_resource_group.agentvault.name
  location           = local.location
  environment        = local.environment
  
  # Network configuration
  compute_subnet_id = module.networking.compute_subnet_id
  
  # Storage integration
  anf_volumes = module.storage.anf_volumes
  
  # Security integration
  key_vault_id        = module.security.key_vault_id
  managed_identity_id = module.security.compute_managed_identity_id
  
  # Monitoring integration
  log_analytics_workspace_id = module.monitoring.log_analytics_workspace_id
  application_insights_key   = module.monitoring.application_insights_key
  
  tags = local.common_tags
  
  depends_on = [module.storage, module.security, module.monitoring]
}

# Redis Cache for high-performance caching
resource "azurerm_redis_cache" "agentvault" {
  name                = "${local.name_prefix}-redis"
  location            = local.location
  resource_group_name = azurerm_resource_group.agentvault.name
  
  capacity            = var.redis_capacity
  family              = var.redis_family
  sku_name           = var.redis_sku
  
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"
  
  # Enterprise features
  zones = var.redis_zones
  
  # Backup configuration
  redis_configuration {
    enable_authentication           = true
    maxmemory_reserved             = 125
    maxmemory_delta                = 125
    maxmemory_policy               = "allkeys-lru"
    maxfragmentationmemory_reserved = 125
    rdb_backup_enabled             = true
    rdb_backup_frequency           = 60
    rdb_backup_max_snapshot_count  = 1
    rdb_storage_connection_string  = module.storage.backup_storage_connection_string
  }
  
  # Network security
  subnet_id = module.networking.compute_subnet_id
  
  tags = local.common_tags
}

# Outputs
output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.agentvault.name
}

output "location" {
  description = "Azure region where resources are deployed"
  value       = local.location
}

output "netapp_account_name" {
  description = "Name of the Azure NetApp Files account"
  value       = module.storage.netapp_account_name
}

output "anf_volumes" {
  description = "Created ANF volumes with mount information"
  value       = module.storage.anf_volumes
  sensitive   = false
}

output "key_vault_uri" {
  description = "URI of the Key Vault for secrets management"
  value       = module.security.key_vault_uri
}

output "redis_connection_string" {
  description = "Redis cache connection string"
  value       = azurerm_redis_cache.agentvault.primary_connection_string
  sensitive   = true
}

output "monitoring_endpoints" {
  description = "Monitoring and observability endpoints"
  value = {
    application_insights = module.monitoring.application_insights_connection_string
    log_analytics       = module.monitoring.log_analytics_workspace_id
  }
  sensitive = true
}

output "compute_resources" {
  description = "Deployed compute resources for AI agents"
  value       = module.compute.deployed_resources
}

output "deployment_summary" {
  description = "Complete deployment summary"
  value = {
    project_name     = local.project_name
    environment      = local.environment
    deployment_time  = timestamp()
    resource_count   = length(keys(local.common_tags))
    estimated_cost   = "Contact support for cost estimation"
    documentation    = "https://github.com/DwirefS/AgentVault/blob/main/README.md"
    support_contact  = "DwirefS@SapientEdge.io"
  }
}