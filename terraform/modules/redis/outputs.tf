# AgentVault™ Redis Cache Module - Outputs
# ========================================
# Exported values for integration with other modules and applications
#
# These outputs provide:
# - Connection strings and endpoints for each cache tier
# - Authentication credentials and Key Vault references
# - Monitoring and diagnostic resource IDs
# - Circuit breaker and performance configurations
# - Private endpoint IP addresses for secure access
#
# Author: Dwiref Sharma
# Contact: DwirefS@SapientEdge.io

# Redis Instance Information
output "redis_instances" {
  description = "Detailed information about each Redis cache instance"
  value = {
    for tier, cache in azurerm_redis_cache.cache_instances : tier => {
      # Basic information
      id               = cache.id
      name             = cache.name
      resource_group   = cache.resource_group_name
      location         = cache.location
      
      # Connection details
      hostname         = cache.hostname
      ssl_port         = cache.ssl_port
      non_ssl_port     = cache.port
      
      # Configuration
      sku_name         = cache.sku_name
      family           = cache.family
      capacity         = cache.capacity
      shard_count      = cache.shard_count
      redis_version    = cache.redis_version
      
      # Network endpoints
      primary_endpoint = var.enable_private_endpoint ? 
        azurerm_private_endpoint.redis_endpoints[tier].private_service_connection[0].private_ip_address : 
        cache.hostname
      
      # Features
      zone_redundant   = cache.zones != null
      clustering_enabled = cache.shard_count > 0
    }
  }
}

# Connection Strings
output "redis_connection_strings" {
  description = "Redis connection strings for each tier (password placeholder)"
  value = {
    for tier, cache in azurerm_redis_cache.cache_instances : tier => {
      # Standard connection string
      standard = var.enable_private_endpoint ?
        "${azurerm_private_endpoint.redis_endpoints[tier].private_service_connection[0].private_ip_address}:${cache.ssl_port},password=${KEY_VAULT_SECRET_NAME},ssl=True,abortConnect=False,connectTimeout=10000,syncTimeout=10000" :
        "${cache.hostname}:${cache.ssl_port},password=${KEY_VAULT_SECRET_NAME},ssl=True,abortConnect=False,connectTimeout=10000,syncTimeout=10000"
      
      # StackExchange.Redis format
      stackexchange = var.enable_private_endpoint ?
        "${azurerm_private_endpoint.redis_endpoints[tier].private_service_connection[0].private_ip_address}:${cache.ssl_port},password=${KEY_VAULT_SECRET_NAME},ssl=True,abortConnect=False,connectTimeout=10000,syncTimeout=10000,allowAdmin=False,connectRetry=3,responseTimeout=5000" :
        "${cache.hostname}:${cache.ssl_port},password=${KEY_VAULT_SECRET_NAME},ssl=True,abortConnect=False,connectTimeout=10000,syncTimeout=10000,allowAdmin=False,connectRetry=3,responseTimeout=5000"
      
      # Key Vault secret reference
      password_secret_name = azurerm_key_vault_secret.redis_password.name
    }
  }
  sensitive = true
}

# Authentication
output "redis_auth" {
  description = "Redis authentication information"
  value = {
    password_key_vault_secret_id   = azurerm_key_vault_secret.redis_password.id
    password_key_vault_secret_name = azurerm_key_vault_secret.redis_password.name
    password_key_vault_uri        = azurerm_key_vault_secret.redis_password.versionless_id
  }
}

# Private Endpoints
output "private_endpoints" {
  description = "Private endpoint information for secure access"
  value = var.enable_private_endpoint ? {
    for tier, endpoint in azurerm_private_endpoint.redis_endpoints : tier => {
      id         = endpoint.id
      name       = endpoint.name
      ip_address = endpoint.private_service_connection[0].private_ip_address
      fqdn       = endpoint.custom_dns_configs[0].fqdn
    }
  } : {}
}

# DNS Configuration
output "dns_configuration" {
  description = "DNS zone configuration for Redis"
  value = {
    private_dns_zone_id   = azurerm_private_dns_zone.redis_dns.id
    private_dns_zone_name = azurerm_private_dns_zone.redis_dns.name
    dns_zone_link_id     = azurerm_private_dns_zone_virtual_network_link.redis_dns_link.id
  }
}

# Redis Enterprise
output "redis_enterprise" {
  description = "Redis Enterprise cluster information"
  value = var.enable_redis_enterprise ? {
    cluster_id   = azurerm_redis_enterprise_cluster.ultra_cache[0].id
    hostname     = azurerm_redis_enterprise_cluster.ultra_cache[0].hostname
    database_id  = azurerm_redis_enterprise_database.ultra_db[0].id
    database_endpoint = "${azurerm_redis_enterprise_cluster.ultra_cache[0].hostname}:10000"
  } : null
}

# Monitoring
output "monitoring_resources" {
  description = "Monitoring resource IDs"
  value = {
    diagnostic_settings = {
      for tier in keys(local.cache_tiers) : tier => azurerm_monitor_diagnostic_setting.redis_diagnostics[tier].id
    }
    metric_alerts = {
      for tier in keys(local.cache_tiers) : tier => azurerm_monitor_metric_alert.redis_alerts[tier].id
    }
  }
}

# Application Configuration
output "application_config" {
  description = "Configuration for application integration"
  value = {
    # Connection pool settings
    connection_pool = {
      min_size = 10
      max_size = 50
      timeout  = 10000
    }
    
    # Circuit breaker configuration
    circuit_breaker = {
      enabled                = true
      failure_threshold      = 5
      success_threshold      = 2
      timeout_seconds        = 30
      half_open_max_requests = 3
    }
    
    # Retry policy
    retry_policy = {
      max_attempts     = 3
      initial_interval = 1000
      max_interval     = 10000
      multiplier       = 2
    }
    
    # Cache TTL defaults (seconds)
    default_ttl = {
      vectors = 3600   # 1 hour
      memory  = 86400  # 24 hours
      api     = 300    # 5 minutes
      session = 7200   # 2 hours
    }
    
    # Eviction policies
    eviction_policies = {
      for tier, config in local.cache_tiers : tier => config.eviction_policy
    }
  }
}

# Performance Metrics
output "performance_config" {
  description = "Performance configuration and limits"
  value = {
    for tier, cache in azurerm_redis_cache.cache_instances : tier => {
      # Memory limits
      total_memory_gb    = cache.family == "P" ? 
        (cache.capacity == 1 ? 6 : 
         cache.capacity == 2 ? 13 : 
         cache.capacity == 3 ? 26 : 
         cache.capacity == 4 ? 53 : 
         cache.capacity == 5 ? 120 : 0) : 0
      
      # Performance characteristics
      max_clients        = 40000
      max_connections    = cache.family == "P" ? cache.capacity * 10000 : 10000
      bandwidth_mbps     = cache.family == "P" ? cache.capacity * 1000 : 500
      
      # Shard configuration
      shards            = cache.shard_count
      nodes             = cache.shard_count > 0 ? cache.shard_count * 2 : 2  # Primary + replica
      
      # Expected latency (milliseconds)
      expected_latency = {
        read_avg  = 0.5
        write_avg = 1.0
        p99       = 5.0
      }
    }
  }
}

# Cost Estimation
output "cost_estimation" {
  description = "Estimated monthly costs for Redis instances"
  value = {
    for tier, cache in azurerm_redis_cache.cache_instances : tier => {
      sku              = "${cache.family}${cache.capacity}"
      estimated_cost   = cache.family == "P" ? 
        (cache.capacity * 250 * (cache.shard_count > 0 ? cache.shard_count : 1)) : 
        100  # Rough estimates in USD
      
      cost_optimization_tips = [
        "Use appropriate eviction policies",
        "Enable compression where possible",
        "Monitor cache hit rates",
        "Right-size based on actual usage"
      ]
    }
  }
}

# Integration Examples
output "integration_examples" {
  description = "Code examples for integrating with Redis"
  value = {
    python = <<-EOT
      import redis
      from azure.identity import DefaultAzureCredential
      from azure.keyvault.secrets import SecretClient
      
      # Get password from Key Vault
      credential = DefaultAzureCredential()
      client = SecretClient(vault_url="${var.key_vault_id}", credential=credential)
      password = client.get_secret("${azurerm_key_vault_secret.redis_password.name}").value
      
      # Connect to Redis
      r = redis.Redis(
          host='${var.enable_private_endpoint ? "PRIVATE_IP" : "HOSTNAME"}',
          port=6380,
          password=password,
          ssl=True,
          ssl_cert_reqs=None,
          decode_responses=True
      )
    EOT
    
    csharp = <<-EOT
      using StackExchange.Redis;
      using Azure.Identity;
      using Azure.Security.KeyVault.Secrets;
      
      // Get password from Key Vault
      var credential = new DefaultAzureCredential();
      var client = new SecretClient(new Uri("${var.key_vault_id}"), credential);
      var password = client.GetSecret("${azurerm_key_vault_secret.redis_password.name}").Value.Value;
      
      // Connect to Redis
      var connection = ConnectionMultiplexer.Connect($"${var.enable_private_endpoint ? "PRIVATE_IP" : "HOSTNAME"}:6380,password={password},ssl=true,abortConnect=false");
    EOT
  }
}