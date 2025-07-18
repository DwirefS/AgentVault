# AgentVault™ AKS Module
# Production-grade Azure Kubernetes Service with HA, security, and monitoring

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

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28.3"
}

variable "sku_tier" {
  description = "SKU tier (Free or Standard)"
  type        = string
  default     = "Standard"
}

variable "system_node_pool" {
  description = "System node pool configuration"
  type = object({
    name               = string
    node_count         = number
    min_count          = number
    max_count          = number
    vm_size            = string
    availability_zones = list(string)
    max_pods           = number
    os_disk_size_gb    = number
    os_disk_type       = string
    ultra_ssd_enabled  = bool
  })
}

variable "user_node_pools" {
  description = "List of user node pools"
  type = list(object({
    name               = string
    node_count         = number
    min_count          = number
    max_count          = number
    vm_size            = string
    availability_zones = list(string)
    max_pods           = number
    os_disk_size_gb    = number
    os_disk_type       = string
    ultra_ssd_enabled  = bool
    node_labels        = map(string)
    node_taints        = list(string)
  }))
  default = []
}

variable "vnet_subnet_id" {
  description = "Subnet ID for AKS nodes"
  type        = string
}

variable "network_plugin" {
  description = "Network plugin (azure or kubenet)"
  type        = string
  default     = "azure"
}

variable "network_policy" {
  description = "Network policy (calico or azure)"
  type        = string
  default     = "calico"
}

variable "service_cidr" {
  description = "Service CIDR"
  type        = string
  default     = "10.100.0.0/16"
}

variable "dns_service_ip" {
  description = "DNS service IP"
  type        = string
  default     = "10.100.0.10"
}

variable "enable_pod_security_policy" {
  description = "Enable pod security policy"
  type        = bool
  default     = true
}

variable "enable_azure_policy" {
  description = "Enable Azure Policy addon"
  type        = bool
  default     = true
}

variable "enable_workload_identity" {
  description = "Enable workload identity"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable monitoring"
  type        = bool
  default     = true
}

variable "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID"
  type        = string
}

variable "enable_ingress_application_gateway" {
  description = "Enable Application Gateway Ingress Controller"
  type        = bool
  default     = true
}

variable "enable_key_vault_secrets_provider" {
  description = "Enable Key Vault secrets provider"
  type        = bool
  default     = true
}

variable "key_vault_id" {
  description = "Key Vault ID for secrets provider"
  type        = string
  default     = ""
}

variable "rbac_azure_ad_managed" {
  description = "Enable Azure AD managed RBAC"
  type        = bool
  default     = true
}

variable "rbac_azure_ad_admin_group_ids" {
  description = "Azure AD admin group IDs"
  type        = list(string)
  default     = []
}

variable "enable_backup" {
  description = "Enable AKS backup"
  type        = bool
  default     = true
}

variable "identity_ids" {
  description = "User assigned identity IDs"
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = "${var.name_prefix}-aks"
  location            = var.location
  resource_group_name = var.resource_group_name
  dns_prefix          = var.name_prefix
  kubernetes_version  = var.kubernetes_version
  sku_tier            = var.sku_tier
  
  # Automatic upgrades
  automatic_channel_upgrade = "stable"
  
  # System node pool
  default_node_pool {
    name                         = var.system_node_pool.name
    node_count                   = var.system_node_pool.node_count
    min_count                    = var.system_node_pool.min_count
    max_count                    = var.system_node_pool.max_count
    vm_size                      = var.system_node_pool.vm_size
    vnet_subnet_id               = var.vnet_subnet_id
    zones                        = var.system_node_pool.availability_zones
    enable_auto_scaling          = true
    max_pods                     = var.system_node_pool.max_pods
    os_disk_size_gb              = var.system_node_pool.os_disk_size_gb
    os_disk_type                 = var.system_node_pool.os_disk_type
    ultra_ssd_enabled            = var.system_node_pool.ultra_ssd_enabled
    only_critical_addons_enabled = true
    
    # Node configuration
    node_labels = {
      "nodepool-type"    = "system"
      "environment"      = var.tags["Environment"]
      "kubernetes.io/os" = "linux"
    }
    
    # Upgrade settings
    upgrade_settings {
      max_surge = "33%"
    }
  }
  
  # Identity
  identity {
    type         = "UserAssigned"
    identity_ids = var.identity_ids
  }
  
  # Network configuration
  network_profile {
    network_plugin    = var.network_plugin
    network_policy    = var.network_policy
    load_balancer_sku = "standard"
    outbound_type     = "loadBalancer"
    service_cidr      = var.service_cidr
    dns_service_ip    = var.dns_service_ip
    
    load_balancer_profile {
      outbound_ports_allocated  = 0
      idle_timeout_in_minutes   = 30
      managed_outbound_ip_count = 2
    }
  }
  
  # RBAC and Azure AD
  azure_active_directory_role_based_access_control {
    managed                = var.rbac_azure_ad_managed
    azure_rbac_enabled     = true
    admin_group_object_ids = var.rbac_azure_ad_admin_group_ids
  }
  
  # Workload Identity
  workload_identity_enabled = var.enable_workload_identity
  oidc_issuer_enabled      = var.enable_workload_identity
  
  # API server access
  api_server_access_profile {
    authorized_ip_ranges = []  # Configure based on security requirements
  }
  
  # Auto-scaler profile
  auto_scaler_profile {
    balance_similar_node_groups      = true
    expander                         = "random"
    max_graceful_termination_sec     = 600
    max_node_provisioning_time       = "15m"
    max_unready_nodes                = 3
    max_unready_percentage           = 45
    new_pod_scale_up_delay           = "10s"
    scale_down_delay_after_add       = "10m"
    scale_down_delay_after_delete    = "10s"
    scale_down_delay_after_failure   = "3m"
    scan_interval                    = "10s"
    scale_down_unneeded              = "10m"
    scale_down_unready               = "20m"
    scale_down_utilization_threshold = "0.5"
    empty_bulk_delete_max            = 10
    skip_nodes_with_local_storage    = true
    skip_nodes_with_system_pods      = true
  }
  
  # Maintenance window
  maintenance_window {
    allowed {
      day   = "Sunday"
      hours = [2, 6]
    }
  }
  
  # Monitoring addon
  dynamic "oms_agent" {
    for_each = var.enable_monitoring ? [1] : []
    content {
      log_analytics_workspace_id = var.log_analytics_workspace_id
    }
  }
  
  # Key Vault secrets provider addon
  dynamic "key_vault_secrets_provider" {
    for_each = var.enable_key_vault_secrets_provider ? [1] : []
    content {
      secret_rotation_enabled  = true
      secret_rotation_interval = "2m"
    }
  }
  
  # Application Gateway ingress controller addon
  dynamic "ingress_application_gateway" {
    for_each = var.enable_ingress_application_gateway ? [1] : []
    content {
      gateway_name = "${var.name_prefix}-appgw"
      subnet_id    = var.vnet_subnet_id
    }
  }
  
  # Azure Policy addon
  azure_policy_enabled = var.enable_azure_policy
  
  # HTTP application routing (disabled for production)
  http_application_routing_enabled = false
  
  # Service mesh addon
  service_mesh_profile {
    mode = "Istio"
    internal_ingress_gateway_enabled = true
    external_ingress_gateway_enabled = true
  }
  
  # Workload autoscaler
  workload_autoscaler_profile {
    keda_enabled = true
  }
  
  # Storage profile
  storage_profile {
    blob_driver_enabled         = true
    disk_driver_enabled         = true
    file_driver_enabled         = true
    snapshot_controller_enabled = true
  }
  
  tags = var.tags
}

# User Node Pools
resource "azurerm_kubernetes_cluster_node_pool" "user" {
  for_each = { for pool in var.user_node_pools : pool.name => pool }
  
  name                  = each.value.name
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = each.value.vm_size
  node_count            = each.value.node_count
  min_count             = each.value.min_count
  max_count             = each.value.max_count
  zones                 = each.value.availability_zones
  enable_auto_scaling   = true
  max_pods              = each.value.max_pods
  os_disk_size_gb       = each.value.os_disk_size_gb
  os_disk_type          = each.value.os_disk_type
  ultra_ssd_enabled     = each.value.ultra_ssd_enabled
  vnet_subnet_id        = var.vnet_subnet_id
  
  node_labels = each.value.node_labels
  node_taints = each.value.node_taints
  
  upgrade_settings {
    max_surge = "33%"
  }
  
  tags = var.tags
}

# Application Gateway for Ingress
resource "azurerm_application_gateway" "main" {
  count               = var.enable_ingress_application_gateway ? 1 : 0
  name                = "${var.name_prefix}-appgw"
  location            = var.location
  resource_group_name = var.resource_group_name
  
  sku {
    name     = "WAF_v2"
    tier     = "WAF_v2"
    capacity = 2
  }
  
  gateway_ip_configuration {
    name      = "gateway-ip-config"
    subnet_id = var.vnet_subnet_id
  }
  
  frontend_port {
    name = "http"
    port = 80
  }
  
  frontend_port {
    name = "https"
    port = 443
  }
  
  frontend_ip_configuration {
    name                 = "frontend-ip"
    public_ip_address_id = azurerm_public_ip.appgw[0].id
  }
  
  backend_address_pool {
    name = "default-backend-pool"
  }
  
  backend_http_settings {
    name                  = "default-http-settings"
    cookie_based_affinity = "Disabled"
    port                  = 80
    protocol              = "Http"
    request_timeout       = 30
  }
  
  http_listener {
    name                           = "default-http-listener"
    frontend_ip_configuration_name = "frontend-ip"
    frontend_port_name             = "http"
    protocol                       = "Http"
  }
  
  request_routing_rule {
    name                       = "default-routing-rule"
    rule_type                  = "Basic"
    http_listener_name         = "default-http-listener"
    backend_address_pool_name  = "default-backend-pool"
    backend_http_settings_name = "default-http-settings"
    priority                   = 100
  }
  
  waf_configuration {
    enabled          = true
    firewall_mode    = "Prevention"
    rule_set_type    = "OWASP"
    rule_set_version = "3.2"
  }
  
  autoscale_configuration {
    min_capacity = 2
    max_capacity = 10
  }
  
  zones = ["1", "2", "3"]
  
  tags = var.tags
}

# Public IP for Application Gateway
resource "azurerm_public_ip" "appgw" {
  count               = var.enable_ingress_application_gateway ? 1 : 0
  name                = "${var.name_prefix}-appgw-pip"
  location            = var.location
  resource_group_name = var.resource_group_name
  allocation_method   = "Static"
  sku                 = "Standard"
  zones               = ["1", "2", "3"]
  
  tags = var.tags
}

# Container Registry
resource "azurerm_container_registry" "main" {
  name                = replace("${var.name_prefix}acr", "-", "")
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = "Premium"
  admin_enabled       = false
  
  # Geo-replication
  georeplications {
    location                = "westus"
    zone_redundancy_enabled = true
    tags                    = var.tags
  }
  
  # Network rules
  network_rule_set {
    default_action = "Allow"  # Configure based on security requirements
  }
  
  # Retention policy
  retention_policy {
    days    = 30
    enabled = true
  }
  
  # Trust policy
  trust_policy {
    enabled = true
  }
  
  # Encryption
  encryption {
    enabled            = true
    key_vault_key_id   = var.key_vault_id
    identity_client_id = var.identity_ids[0]
  }
  
  # Zone redundancy
  zone_redundancy_enabled = true
  
  tags = var.tags
}

# Role assignment for AKS to pull images from ACR
resource "azurerm_role_assignment" "aks_acr_pull" {
  principal_id                     = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                            = azurerm_container_registry.main.id
  skip_service_principal_aad_check = true
}

# Outputs
output "cluster_id" {
  value = azurerm_kubernetes_cluster.main.id
}

output "cluster_name" {
  value = azurerm_kubernetes_cluster.main.name
}

output "cluster_fqdn" {
  value = azurerm_kubernetes_cluster.main.fqdn
}

output "kube_config" {
  value     = azurerm_kubernetes_cluster.main.kube_admin_config
  sensitive = true
}

output "kube_config_raw" {
  value     = azurerm_kubernetes_cluster.main.kube_admin_config_raw
  sensitive = true
}

output "node_resource_group" {
  value = azurerm_kubernetes_cluster.main.node_resource_group
}

output "oidc_issuer_url" {
  value = azurerm_kubernetes_cluster.main.oidc_issuer_url
}

output "kubelet_identity" {
  value = azurerm_kubernetes_cluster.main.kubelet_identity[0]
}

output "ingress_ip" {
  value = var.enable_ingress_application_gateway ? azurerm_public_ip.appgw[0].ip_address : ""
}

output "acr_login_server" {
  value = azurerm_container_registry.main.login_server
}