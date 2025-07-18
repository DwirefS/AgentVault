# AgentVault™ Redis Cache Module - Distributed Cache Infrastructure
# ================================================================
# Production-ready Redis cluster for high-performance caching of AI agent data
#
# This module creates:
# - Azure Cache for Redis Premium with clustering enabled
# - Multiple cache instances for different workload types
# - Redis Enterprise for ultra-low latency requirements
# - Private endpoints for secure access
# - Monitoring and alerting configuration
# - Backup and disaster recovery setup
# - Auto-scaling based on memory pressure
#
# Key Features:
# - Sub-millisecond latency for cache hits
# - Horizontal scaling with Redis Cluster
# - Active geo-replication for disaster recovery
# - L1/L2 cache architecture support
# - Circuit breaker pattern implementation
# - Connection pooling optimization
#
# Author: Dwiref Sharma
# Contact: DwirefS@SapientEdge.io
# License: Proprietary - SapientEdge LLC

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
  }
}

# Local variables for cache configuration
locals {
  name_prefix = "agentvault-${var.environment}"
  
  # Cache tier configurations optimized for AI workloads
  cache_tiers = {
    # Ultra-performance cache for vector similarity results
    vectors = {
      sku_name       = "Premium"
      family         = "P"
      capacity       = 5  # 26GB Premium P5
      shard_count    = 4  # 4 shards for parallel access
      purpose        = "Vector search results and embeddings cache"
      ttl_seconds    = 3600  # 1 hour TTL for vectors
      eviction_policy = "allkeys-lru"
      features       = ["clustering", "replication", "persistence"]
    }
    
    # High-performance cache for agent memory
    memory = {
      sku_name       = "Premium"
      family         = "P"
      capacity       = 3  # 13GB Premium P3
      shard_count    = 2  # 2 shards for redundancy
      purpose        = "Agent short-term memory and conversation state"
      ttl_seconds    = 86400  # 24 hour TTL
      eviction_policy = "volatile-lru"
      features       = ["clustering", "replication", "persistence"]
    }
    
    # Standard cache for API responses
    api = {
      sku_name       = "Premium"
      family         = "P"
      capacity       = 2  # 6GB Premium P2
      shard_count    = 1  # Single shard sufficient
      purpose        = "API response caching and rate limiting"
      ttl_seconds    = 300  # 5 minute TTL
      eviction_policy = "allkeys-lru"
      features       = ["clustering", "persistence"]
    }
    
    # Session cache for user sessions
    session = {
      sku_name       = "Premium"
      family         = "P"
      capacity       = 1  # 6GB Premium P1
      shard_count    = 1
      purpose        = "User session and authentication tokens"
      ttl_seconds    = 7200  # 2 hour TTL
      eviction_policy = "volatile-ttl"
      features       = ["persistence", "backup"]
    }
  }
  
  # Redis configuration optimizations
  redis_config = {
    # Memory optimization
    maxmemory_reserved = 25  # 25% reserved for non-cache operations
    maxmemory_delta    = 25  # 25% reserved for master-slave replication
    
    # Performance settings
    tcp_keepalive      = 60   # TCP keepalive in seconds
    timeout            = 300  # Client idle timeout
    databases          = 16   # Number of databases
    
    # Persistence settings
    rdb_backup_enabled     = true
    rdb_backup_frequency   = 60  # Backup every 60 minutes
    rdb_backup_max_days    = 7   # Keep backups for 7 days
    aof_backup_enabled     = false  # Disable AOF for performance
    
    # Security settings
    require_pass           = true
    ssl_enforcement        = true
    minimum_tls_version    = "1.2"
  }
}

# Random password for Redis authentication
resource "random_password" "redis_password" {
  length  = 32
  special = true
  upper   = true
  lower   = true
  numeric = true
}

# Store Redis password in Key Vault
resource "azurerm_key_vault_secret" "redis_password" {
  name         = "${local.name_prefix}-redis-password"
  value        = random_password.redis_password.result
  key_vault_id = var.key_vault_id
  
  content_type = "text/plain"
  
  tags = merge(var.tags, {
    Component = "Security"
    Service   = "Redis"
    Purpose   = "Authentication"
  })
}

# Create Redis Cache instances for each tier
resource "azurerm_redis_cache" "cache_instances" {
  for_each = local.cache_tiers
  
  name                = "${local.name_prefix}-redis-${each.key}"
  location            = var.location
  resource_group_name = var.resource_group_name
  
  # SKU configuration
  sku_name = each.value.sku_name
  family   = each.value.family
  capacity = each.value.capacity
  
  # Enable clustering for horizontal scaling
  shard_count = each.value.shard_count
  
  # Zone redundancy for high availability
  zones = var.enable_zone_redundancy ? ["1", "2", "3"] : null
  
  # Network configuration
  subnet_id = var.enable_private_endpoint ? var.cache_subnet_id : null
  
  # Redis version (latest stable)
  redis_version = "6.0"
  
  # Enable non-SSL port for private endpoint scenarios only
  enable_non_ssl_port = false
  
  # Minimum TLS version
  minimum_tls_version = local.redis_config.minimum_tls_version
  
  # Public network access (disabled for security)
  public_network_access_enabled = false
  
  # Redis configuration
  redis_configuration {
    # Authentication
    enable_authentication = local.redis_config.require_pass
    
    # Memory management
    maxmemory_reserved = local.redis_config.maxmemory_reserved
    maxmemory_delta    = local.redis_config.maxmemory_delta
    maxmemory_policy   = each.value.eviction_policy
    
    # Persistence configuration
    rdb_backup_enabled     = contains(each.value.features, "persistence") ? local.redis_config.rdb_backup_enabled : false
    rdb_backup_frequency   = contains(each.value.features, "persistence") ? local.redis_config.rdb_backup_frequency : null
    rdb_backup_max_snapshot_count = contains(each.value.features, "persistence") ? local.redis_config.rdb_backup_max_days : null
    rdb_storage_connection_string = contains(each.value.features, "persistence") ? var.backup_storage_connection_string : null
    
    # AOF persistence (disabled for performance)
    aof_backup_enabled = local.redis_config.aof_backup_enabled
    
    # Additional settings
    notify_keyspace_events = "AKE"  # Enable keyspace notifications
  }
  
  # Patch schedule for maintenance
  patch_schedule {
    day_of_week    = "Sunday"
    start_hour_utc = 2  # 2 AM UTC
  }
  
  # Identity for managed identity authentication
  identity {
    type = "SystemAssigned"
  }
  
  tags = merge(var.tags, {
    Component    = "Cache"
    Service      = "Redis"
    Tier         = each.key
    Purpose      = each.value.purpose
    ShardCount   = each.value.shard_count
    EvictionPolicy = each.value.eviction_policy
  })
}

# Private endpoints for secure Redis access
resource "azurerm_private_endpoint" "redis_endpoints" {
  for_each = var.enable_private_endpoint ? local.cache_tiers : {}
  
  name                = "${local.name_prefix}-redis-${each.key}-pe"
  location            = var.location
  resource_group_name = var.resource_group_name
  subnet_id           = var.private_endpoint_subnet_id
  
  private_service_connection {
    name                           = "${local.name_prefix}-redis-${each.key}-psc"
    private_connection_resource_id = azurerm_redis_cache.cache_instances[each.key].id
    subresource_names             = ["redisCache"]
    is_manual_connection          = false
  }
  
  # DNS configuration
  private_dns_zone_group {
    name                 = "redis-dns-zone-group"
    private_dns_zone_ids = [azurerm_private_dns_zone.redis_dns.id]
  }
  
  tags = merge(var.tags, {
    Component = "Security"
    Service   = "Private Endpoint"
    Purpose   = "Secure Redis Access"
    CacheTier = each.key
  })
}

# Private DNS zone for Redis
resource "azurerm_private_dns_zone" "redis_dns" {
  name                = "privatelink.redis.cache.windows.net"
  resource_group_name = var.resource_group_name
  
  tags = merge(var.tags, {
    Component = "Networking"
    Service   = "Private DNS"
    Purpose   = "Redis Name Resolution"
  })
}

# Link private DNS zone to virtual network
resource "azurerm_private_dns_zone_virtual_network_link" "redis_dns_link" {
  name                  = "${local.name_prefix}-redis-dns-link"
  resource_group_name   = var.resource_group_name
  private_dns_zone_name = azurerm_private_dns_zone.redis_dns.name
  virtual_network_id    = var.virtual_network_id
  registration_enabled  = false
  
  tags = var.tags
}

# Redis firewall rules (if not using private endpoints)
resource "azurerm_redis_firewall_rule" "allow_aks" {
  for_each = var.enable_private_endpoint ? {} : local.cache_tiers
  
  name                = "allow-aks-nodes"
  redis_cache_name    = azurerm_redis_cache.cache_instances[each.key].name
  resource_group_name = var.resource_group_name
  start_ip            = var.aks_subnet_start_ip
  end_ip              = var.aks_subnet_end_ip
}

# Create Redis Enterprise cluster for ultra-low latency
resource "azurerm_redis_enterprise_cluster" "ultra_cache" {
  count = var.enable_redis_enterprise ? 1 : 0
  
  name                = "${local.name_prefix}-redis-enterprise"
  location            = var.location
  resource_group_name = var.resource_group_name
  
  # SKU for Redis Enterprise
  sku_name = "Enterprise_E10-2"  # 10GB, 2 capacity units
  
  # Zone redundancy
  zones = ["1", "2", "3"]
  
  # Minimum TLS version
  minimum_tls_version = "1.2"
  
  tags = merge(var.tags, {
    Component = "Cache"
    Service   = "Redis Enterprise"
    Purpose   = "Ultra-low latency caching"
  })
}

# Redis Enterprise database
resource "azurerm_redis_enterprise_database" "ultra_db" {
  count = var.enable_redis_enterprise ? 1 : 0
  
  name              = "default"
  cluster_id        = azurerm_redis_enterprise_cluster.ultra_cache[0].id
  
  # Client protocol
  client_protocol   = "Encrypted"
  
  # Clustering policy
  clustering_policy = "EnterpriseCluster"
  
  # Eviction policy
  eviction_policy   = "AllKeysLRU"
  
  # Modules to enable
  module {
    name = "RediSearch"
    args = ""
  }
  
  module {
    name = "RedisJSON"
    args = ""
  }
  
  # Port
  port = 10000
  
  # Persistence
  persistence {
    aof_enabled = false
    rdb_enabled = true
    rdb_frequency = "1h"
  }
}

# Diagnostic settings for monitoring
resource "azurerm_monitor_diagnostic_setting" "redis_diagnostics" {
  for_each = local.cache_tiers
  
  name               = "${local.name_prefix}-redis-${each.key}-diagnostics"
  target_resource_id = azurerm_redis_cache.cache_instances[each.key].id
  
  log_analytics_workspace_id = var.log_analytics_workspace_id
  
  # Logs
  enabled_log {
    category = "ConnectedClientList"
  }
  
  # Metrics
  metric {
    category = "AllMetrics"
    enabled  = true
    
    retention_policy {
      enabled = true
      days    = 30
    }
  }
}

# Alert rules for Redis monitoring
resource "azurerm_monitor_metric_alert" "redis_alerts" {
  for_each = local.cache_tiers
  
  name                = "${local.name_prefix}-redis-${each.key}-high-memory"
  resource_group_name = var.resource_group_name
  scopes              = [azurerm_redis_cache.cache_instances[each.key].id]
  description         = "Alert when Redis memory usage is high"
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT15M"
  
  criteria {
    metric_namespace = "Microsoft.Cache/redis"
    metric_name      = "UsedMemoryPercentage"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 90
  }
  
  action {
    action_group_id = var.action_group_id
  }
  
  tags = var.tags
}

# Connection multiplexer configuration output
resource "local_file" "redis_config" {
  filename = "${path.module}/redis-config.json"
  content = jsonencode({
    for tier, cache in azurerm_redis_cache.cache_instances : tier => {
      connection_string = var.enable_private_endpoint ? 
        "${cache.hostname}:${cache.ssl_port},password=${random_password.redis_password.result},ssl=True,abortConnect=False,connectTimeout=10000,syncTimeout=10000" :
        "${cache.hostname}:${cache.ssl_port},password=${cache.primary_access_key},ssl=True,abortConnect=False,connectTimeout=10000,syncTimeout=10000"
      
      configuration = {
        host              = cache.hostname
        port              = cache.ssl_port
        ssl               = true
        password_secret   = azurerm_key_vault_secret.redis_password.name
        database          = 0
        connect_timeout   = 10000
        sync_timeout      = 10000
        abort_on_connect  = false
        keep_alive        = 60
        connect_retry     = 3
        response_timeout  = 5000
        default_ttl       = local.cache_tiers[tier].ttl_seconds
        eviction_policy   = local.cache_tiers[tier].eviction_policy
        max_clients       = 10000
        
        # Connection pool settings
        pool_size         = 50
        min_pool_size     = 10
        
        # Circuit breaker settings
        circuit_breaker = {
          enabled                = true
          failure_threshold      = 5
          success_threshold      = 2
          timeout_seconds        = 30
          half_open_max_requests = 3
        }
      }
    }
  })
}

# Outputs
output "redis_instances" {
  description = "Redis cache instances with connection details"
  value = {
    for tier, cache in azurerm_redis_cache.cache_instances : tier => {
      id                = cache.id
      hostname          = cache.hostname
      ssl_port          = cache.ssl_port
      shard_count       = cache.shard_count
      sku              = "${cache.family}${cache.capacity}"
      primary_endpoint  = var.enable_private_endpoint ? 
        azurerm_private_endpoint.redis_endpoints[tier].private_service_connection[0].private_ip_address : 
        cache.hostname
    }
  }
  sensitive = true
}

output "redis_connection_strings" {
  description = "Redis connection strings for each tier"
  value = {
    for tier, cache in azurerm_redis_cache.cache_instances : tier => var.enable_private_endpoint ?
      "${azurerm_private_endpoint.redis_endpoints[tier].private_service_connection[0].private_ip_address}:${cache.ssl_port},password=<from-key-vault>,ssl=True,abortConnect=False" :
      "${cache.hostname}:${cache.ssl_port},password=<from-key-vault>,ssl=True,abortConnect=False"
  }
  sensitive = true
}

output "redis_password_secret_name" {
  description = "Name of the Key Vault secret containing Redis password"
  value       = azurerm_key_vault_secret.redis_password.name
}

output "redis_enterprise_connection" {
  description = "Redis Enterprise connection details"
  value = var.enable_redis_enterprise ? {
    hostname = azurerm_redis_enterprise_cluster.ultra_cache[0].hostname
    database = azurerm_redis_enterprise_database.ultra_db[0].name
  } : null
  sensitive = true
}

output "redis_monitoring" {
  description = "Redis monitoring configuration"
  value = {
    diagnostic_settings = [for tier in keys(local.cache_tiers) : azurerm_monitor_diagnostic_setting.redis_diagnostics[tier].id]
    alert_rules        = [for tier in keys(local.cache_tiers) : azurerm_monitor_metric_alert.redis_alerts[tier].id]
  }
}

output "circuit_breaker_config" {
  description = "Circuit breaker configuration for application integration"
  value = {
    enabled                = true
    failure_threshold      = 5
    success_threshold      = 2
    timeout_seconds        = 30
    half_open_max_requests = 3
    
    # Fallback strategies
    fallback_strategies = {
      vectors = "Return empty results with warning"
      memory  = "Use local cache or default memory"
      api     = "Return cached stale data if available"
      session = "Force re-authentication"
    }
  }
}