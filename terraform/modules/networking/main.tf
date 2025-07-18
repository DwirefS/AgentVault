# AgentVault™ Networking Module
# Comprehensive network infrastructure with security and HA

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

variable "vnet_address_space" {
  description = "Address space for virtual network"
  type        = list(string)
}

variable "enable_ddos_protection" {
  description = "Enable DDoS protection"
  type        = bool
  default     = false
}

variable "enable_firewall" {
  description = "Enable Azure Firewall"
  type        = bool
  default     = true
}

variable "enable_bastion" {
  description = "Enable Azure Bastion"
  type        = bool
  default     = true
}

variable "enable_network_watcher" {
  description = "Enable Network Watcher"
  type        = bool
  default     = true
}

variable "enable_flow_logs" {
  description = "Enable NSG flow logs"
  type        = bool
  default     = true
}

variable "create_private_dns_zones" {
  description = "Create private DNS zones"
  type        = bool
  default     = true
}

variable "private_dns_zones" {
  description = "List of private DNS zones to create"
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# DDoS Protection Plan
resource "azurerm_network_ddos_protection_plan" "main" {
  count               = var.enable_ddos_protection ? 1 : 0
  name                = "${var.name_prefix}-ddos"
  location            = var.location
  resource_group_name = var.resource_group_name
  tags                = var.tags
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "${var.name_prefix}-vnet"
  location            = var.location
  resource_group_name = var.resource_group_name
  address_space       = var.vnet_address_space
  
  dynamic "ddos_protection_plan" {
    for_each = var.enable_ddos_protection ? [1] : []
    content {
      id     = azurerm_network_ddos_protection_plan.main[0].id
      enable = true
    }
  }
  
  tags = var.tags
}

# Subnets
resource "azurerm_subnet" "aks" {
  name                 = "${var.name_prefix}-aks-subnet"
  resource_group_name  = var.resource_group_name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_address_space[0], 4, 0)] # /20
  
  service_endpoints = [
    "Microsoft.Storage",
    "Microsoft.KeyVault",
    "Microsoft.ContainerRegistry",
    "Microsoft.Sql"
  ]
}

resource "azurerm_subnet" "anf" {
  name                 = "${var.name_prefix}-anf-subnet"
  resource_group_name  = var.resource_group_name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_address_space[0], 4, 1)] # /20
  
  delegation {
    name = "netapp"
    service_delegation {
      name = "Microsoft.Netapp/volumes"
      actions = [
        "Microsoft.Network/networkinterfaces/*",
        "Microsoft.Network/virtualNetworks/subnets/join/action"
      ]
    }
  }
}

resource "azurerm_subnet" "private_endpoints" {
  name                                          = "${var.name_prefix}-pe-subnet"
  resource_group_name                           = var.resource_group_name
  virtual_network_name                          = azurerm_virtual_network.main.name
  address_prefixes                              = [cidrsubnet(var.vnet_address_space[0], 6, 8)] # /22
  private_endpoint_network_policies_enabled     = false
  private_link_service_network_policies_enabled = false
}

resource "azurerm_subnet" "application_gateway" {
  name                 = "${var.name_prefix}-appgw-subnet"
  resource_group_name  = var.resource_group_name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_address_space[0], 8, 36)] # /24
}

resource "azurerm_subnet" "database" {
  name                 = "${var.name_prefix}-db-subnet"
  resource_group_name  = var.resource_group_name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_address_space[0], 8, 37)] # /24
  
  delegation {
    name = "postgresql"
    service_delegation {
      name = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = [
        "Microsoft.Network/virtualNetworks/subnets/join/action"
      ]
    }
  }
  
  service_endpoints = [
    "Microsoft.Storage"
  ]
}

resource "azurerm_subnet" "redis" {
  name                 = "${var.name_prefix}-redis-subnet"
  resource_group_name  = var.resource_group_name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_address_space[0], 8, 38)] # /24
}

resource "azurerm_subnet" "bastion" {
  count                = var.enable_bastion ? 1 : 0
  name                 = "AzureBastionSubnet" # Required name
  resource_group_name  = var.resource_group_name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_address_space[0], 8, 39)] # /24
}

resource "azurerm_subnet" "firewall" {
  count                = var.enable_firewall ? 1 : 0
  name                 = "AzureFirewallSubnet" # Required name
  resource_group_name  = var.resource_group_name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_address_space[0], 8, 40)] # /24
}

# Network Security Groups
resource "azurerm_network_security_group" "aks" {
  name                = "${var.name_prefix}-aks-nsg"
  location            = var.location
  resource_group_name = var.resource_group_name
  tags                = var.tags
}

resource "azurerm_network_security_group" "anf" {
  name                = "${var.name_prefix}-anf-nsg"
  location            = var.location
  resource_group_name = var.resource_group_name
  tags                = var.tags
}

resource "azurerm_network_security_group" "private_endpoints" {
  name                = "${var.name_prefix}-pe-nsg"
  location            = var.location
  resource_group_name = var.resource_group_name
  tags                = var.tags
}

# NSG Rules for ANF
resource "azurerm_network_security_rule" "anf_nfs" {
  name                        = "AllowNFS"
  priority                    = 100
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range          = "*"
  destination_port_ranges    = ["111", "2049", "635", "4045", "4046", "4049"]
  source_address_prefix      = azurerm_subnet.aks.address_prefixes[0]
  destination_address_prefix = azurerm_subnet.anf.address_prefixes[0]
  resource_group_name        = var.resource_group_name
  network_security_group_name = azurerm_network_security_group.anf.name
}

resource "azurerm_network_security_rule" "anf_smb" {
  name                        = "AllowSMB"
  priority                    = 110
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range          = "*"
  destination_port_range     = "445"
  source_address_prefix      = azurerm_subnet.aks.address_prefixes[0]
  destination_address_prefix = azurerm_subnet.anf.address_prefixes[0]
  resource_group_name        = var.resource_group_name
  network_security_group_name = azurerm_network_security_group.anf.name
}

# NSG Associations
resource "azurerm_subnet_network_security_group_association" "aks" {
  subnet_id                 = azurerm_subnet.aks.id
  network_security_group_id = azurerm_network_security_group.aks.id
}

resource "azurerm_subnet_network_security_group_association" "anf" {
  subnet_id                 = azurerm_subnet.anf.id
  network_security_group_id = azurerm_network_security_group.anf.id
}

resource "azurerm_subnet_network_security_group_association" "private_endpoints" {
  subnet_id                 = azurerm_subnet.private_endpoints.id
  network_security_group_id = azurerm_network_security_group.private_endpoints.id
}

# Azure Firewall
resource "azurerm_public_ip" "firewall" {
  count               = var.enable_firewall ? 1 : 0
  name                = "${var.name_prefix}-fw-pip"
  location            = var.location
  resource_group_name = var.resource_group_name
  allocation_method   = "Static"
  sku                 = "Standard"
  zones               = ["1", "2", "3"]
  tags                = var.tags
}

resource "azurerm_firewall" "main" {
  count               = var.enable_firewall ? 1 : 0
  name                = "${var.name_prefix}-fw"
  location            = var.location
  resource_group_name = var.resource_group_name
  sku_name            = "AZFW_VNet"
  sku_tier            = "Premium"
  zones               = ["1", "2", "3"]
  
  ip_configuration {
    name                 = "configuration"
    subnet_id            = azurerm_subnet.firewall[0].id
    public_ip_address_id = azurerm_public_ip.firewall[0].id
  }
  
  tags = var.tags
}

# Firewall Policy
resource "azurerm_firewall_policy" "main" {
  count               = var.enable_firewall ? 1 : 0
  name                = "${var.name_prefix}-fwpolicy"
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = "Premium"
  
  threat_intelligence_mode = "Alert"
  
  intrusion_detection {
    mode = "Alert"
  }
  
  dns {
    proxy_enabled = true
  }
  
  tags = var.tags
}

# Azure Bastion
resource "azurerm_public_ip" "bastion" {
  count               = var.enable_bastion ? 1 : 0
  name                = "${var.name_prefix}-bastion-pip"
  location            = var.location
  resource_group_name = var.resource_group_name
  allocation_method   = "Static"
  sku                 = "Standard"
  zones               = ["1", "2", "3"]
  tags                = var.tags
}

resource "azurerm_bastion_host" "main" {
  count               = var.enable_bastion ? 1 : 0
  name                = "${var.name_prefix}-bastion"
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = "Standard"
  scale_units         = 2
  
  ip_configuration {
    name                 = "configuration"
    subnet_id            = azurerm_subnet.bastion[0].id
    public_ip_address_id = azurerm_public_ip.bastion[0].id
  }
  
  tags = var.tags
}

# Private DNS Zones
resource "azurerm_private_dns_zone" "main" {
  for_each            = var.create_private_dns_zones ? toset(var.private_dns_zones) : []
  name                = each.value
  resource_group_name = var.resource_group_name
  tags                = var.tags
}

# Link Private DNS Zones to VNet
resource "azurerm_private_dns_zone_virtual_network_link" "main" {
  for_each              = var.create_private_dns_zones ? toset(var.private_dns_zones) : []
  name                  = "${var.name_prefix}-link"
  resource_group_name   = var.resource_group_name
  private_dns_zone_name = azurerm_private_dns_zone.main[each.key].name
  virtual_network_id    = azurerm_virtual_network.main.id
  registration_enabled  = false
  tags                  = var.tags
}

# Network Watcher
resource "azurerm_network_watcher" "main" {
  count               = var.enable_network_watcher ? 1 : 0
  name                = "${var.name_prefix}-nw"
  location            = var.location
  resource_group_name = var.resource_group_name
  tags                = var.tags
}

# Storage Account for Flow Logs
resource "azurerm_storage_account" "flow_logs" {
  count                    = var.enable_flow_logs ? 1 : 0
  name                     = replace("${var.name_prefix}flowlogs", "-", "")
  location                 = var.location
  resource_group_name      = var.resource_group_name
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version         = "TLS1_2"
  
  network_rules {
    default_action = "Deny"
    bypass         = ["AzureServices"]
  }
  
  tags = var.tags
}

# NSG Flow Logs
resource "azurerm_network_watcher_flow_log" "aks" {
  count                    = var.enable_flow_logs && var.enable_network_watcher ? 1 : 0
  name                     = "${var.name_prefix}-aks-flowlog"
  network_watcher_name     = azurerm_network_watcher.main[0].name
  resource_group_name      = var.resource_group_name
  network_security_group_id = azurerm_network_security_group.aks.id
  storage_account_id       = azurerm_storage_account.flow_logs[0].id
  enabled                  = true
  version                  = 2
  
  retention_policy {
    enabled = true
    days    = 30
  }
  
  traffic_analytics {
    enabled = true
  }
  
  tags = var.tags
}

# Route Tables
resource "azurerm_route_table" "main" {
  name                          = "${var.name_prefix}-rt"
  location                      = var.location
  resource_group_name           = var.resource_group_name
  disable_bgp_route_propagation = false
  
  dynamic "route" {
    for_each = var.enable_firewall ? [1] : []
    content {
      name                   = "to-firewall"
      address_prefix         = "0.0.0.0/0"
      next_hop_type          = "VirtualAppliance"
      next_hop_in_ip_address = var.enable_firewall ? azurerm_firewall.main[0].ip_configuration[0].private_ip_address : null
    }
  }
  
  tags = var.tags
}

# Route Table Associations
resource "azurerm_subnet_route_table_association" "aks" {
  count          = var.enable_firewall ? 1 : 0
  subnet_id      = azurerm_subnet.aks.id
  route_table_id = azurerm_route_table.main.id
}

# Outputs
output "vnet_id" {
  value = azurerm_virtual_network.main.id
}

output "vnet_name" {
  value = azurerm_virtual_network.main.name
}

output "aks_subnet_id" {
  value = azurerm_subnet.aks.id
}

output "anf_subnet_id" {
  value = azurerm_subnet.anf.id
}

output "private_endpoint_subnet_id" {
  value = azurerm_subnet.private_endpoints.id
}

output "database_subnet_id" {
  value = azurerm_subnet.database.id
}

output "redis_subnet_id" {
  value = azurerm_subnet.redis.id
}

output "application_gateway_subnet_id" {
  value = azurerm_subnet.application_gateway.id
}

output "firewall_private_ip" {
  value = var.enable_firewall ? azurerm_firewall.main[0].ip_configuration[0].private_ip_address : null
}

output "bastion_fqdn" {
  value = var.enable_bastion ? azurerm_bastion_host.main[0].dns_name : null
}

output "private_dns_zone_ids" {
  value = {
    for k, v in azurerm_private_dns_zone.main : 
    replace(k, "privatelink.", "") => v.id
  }
}

output "network_watcher_id" {
  value = var.enable_network_watcher ? azurerm_network_watcher.main[0].id : null
}

output "flow_logs_storage_account_id" {
  value = var.enable_flow_logs ? azurerm_storage_account.flow_logs[0].id : null
}