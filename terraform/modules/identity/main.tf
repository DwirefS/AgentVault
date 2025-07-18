# AgentVault™ Identity Module
# Azure AD groups, service principals, app registrations, and RBAC

# Variables
variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "create_user_assigned_identity" {
  description = "Create user-assigned managed identity"
  type        = bool
  default     = true
}

variable "identity_name" {
  description = "Name for the managed identity"
  type        = string
  default     = ""
}

variable "create_service_principals" {
  description = "Create service principals"
  type        = bool
  default     = true
}

variable "service_principals" {
  description = "List of service principals to create"
  type = list(object({
    name  = string
    roles = list(string)
  }))
  default = []
}

variable "create_security_groups" {
  description = "Create Azure AD security groups"
  type        = bool
  default     = true
}

variable "security_groups" {
  description = "List of security groups to create"
  type = list(object({
    name        = string
    description = string
    owners      = optional(list(string), [])
    members     = optional(list(string), [])
  }))
  default = []
}

variable "create_app_registrations" {
  description = "Create app registrations"
  type        = bool
  default     = true
}

variable "app_registrations" {
  description = "List of app registrations to create"
  type = list(object({
    name            = string
    api_permissions = list(string)
    redirect_uris   = list(string)
  }))
  default = []
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Data sources
data "azuread_client_config" "current" {}
data "azurerm_subscription" "current" {}

# Random passwords for service principals
resource "random_password" "service_principal" {
  for_each = var.create_service_principals ? { for sp in var.service_principals : sp.name => sp } : {}
  
  length  = 32
  special = true
  
  keepers = {
    service_principal = each.key
  }
}

# Azure AD Applications
resource "azuread_application" "service_principal" {
  for_each = var.create_service_principals ? { for sp in var.service_principals : sp.name => sp } : {}
  
  display_name = each.value.name
  
  feature_tags {
    enterprise = true
  }
  
  owners = [data.azuread_client_config.current.object_id]
}

# Service Principals
resource "azuread_service_principal" "main" {
  for_each = var.create_service_principals ? { for sp in var.service_principals : sp.name => sp } : {}
  
  client_id                    = azuread_application.service_principal[each.key].client_id
  app_role_assignment_required = false
  owners                       = [data.azuread_client_config.current.object_id]
}

# Service Principal Passwords
resource "azuread_service_principal_password" "main" {
  for_each = var.create_service_principals ? { for sp in var.service_principals : sp.name => sp } : {}
  
  service_principal_id = azuread_service_principal.main[each.key].object_id
  
  rotate_when_changed = {
    rotation = timestamp()
  }
}

# Role Assignments for Service Principals
resource "azurerm_role_assignment" "service_principal" {
  for_each = var.create_service_principals ? {
    for item in flatten([
      for sp_key, sp in var.service_principals : [
        for role in sp.roles : {
          key  = "${sp.name}-${role}"
          name = sp.name
          role = role
        }
      ]
    ]) : item.key => item
  } : {}
  
  scope                = data.azurerm_subscription.current.id
  role_definition_name = each.value.role
  principal_id         = azuread_service_principal.main[each.value.name].object_id
}

# Security Groups
resource "azuread_group" "main" {
  for_each = var.create_security_groups ? { for group in var.security_groups : group.name => group } : {}
  
  display_name     = each.value.name
  description      = each.value.description
  security_enabled = true
  
  owners = concat(
    [data.azuread_client_config.current.object_id],
    each.value.owners
  )
  
  dynamic "members" {
    for_each = each.value.members != null ? each.value.members : []
    content {
      # Note: In production, members would be added separately
      # This is a placeholder for the dynamic block structure
    }
  }
}

# App Registrations
resource "azuread_application" "app" {
  for_each = var.create_app_registrations ? { for app in var.app_registrations : app.name => app } : {}
  
  display_name = each.value.name
  
  web {
    redirect_uris = each.value.redirect_uris
    
    implicit_grant {
      access_token_issuance_enabled = true
      id_token_issuance_enabled     = true
    }
  }
  
  api {
    mapped_claims_enabled          = true
    requested_access_token_version = 2
    
    oauth2_permission_scope {
      admin_consent_description  = "Allow the application to access ${each.value.name} on behalf of the signed-in user."
      admin_consent_display_name = "Access ${each.value.name}"
      enabled                    = true
      id                         = random_uuid.oauth2_permission_scope[each.key].result
      type                       = "User"
      user_consent_description   = "Allow the application to access ${each.value.name} on your behalf."
      user_consent_display_name  = "Access ${each.value.name}"
      value                      = "user_impersonation"
    }
  }
  
  required_resource_access {
    resource_app_id = "00000003-0000-0000-c000-000000000000" # Microsoft Graph
    
    dynamic "resource_access" {
      for_each = each.value.api_permissions
      content {
        id   = local.graph_permissions[resource_access.value]
        type = "Delegated"
      }
    }
  }
  
  owners = [data.azuread_client_config.current.object_id]
}

# Service Principals for App Registrations
resource "azuread_service_principal" "app" {
  for_each = var.create_app_registrations ? { for app in var.app_registrations : app.name => app } : {}
  
  client_id                    = azuread_application.app[each.key].client_id
  app_role_assignment_required = false
  owners                       = [data.azuread_client_config.current.object_id]
}

# App Registration Secrets
resource "azuread_application_password" "app" {
  for_each = var.create_app_registrations ? { for app in var.app_registrations : app.name => app } : {}
  
  application_id = azuread_application.app[each.key].id
  display_name   = "${each.value.name}-secret"
  
  rotate_when_changed = {
    rotation = timestamp()
  }
}

# Random UUIDs for OAuth2 permission scopes
resource "random_uuid" "oauth2_permission_scope" {
  for_each = var.create_app_registrations ? { for app in var.app_registrations : app.name => app } : {}
}

# Local variables for Graph API permissions
locals {
  graph_permissions = {
    "User.Read"         = "e1fe6dd8-ba31-4d61-89e7-88639da4683d"
    "Directory.Read.All" = "7ab1d382-f21e-4acd-a863-ba3e13f7da61"
    "Group.Read.All"    = "5b567255-7703-4780-807c-7be8301ae99b"
    "Mail.Send"         = "b633e1c5-b582-4048-a93e-9f11b44c7e96"
  }
}

# Conditional Access Policy (requires Entra ID P1 or P2)
resource "azuread_conditional_access_policy" "mfa" {
  display_name = "${var.name_prefix}-require-mfa"
  state        = "enabledForReportingButNotEnforced" # Start in report-only mode
  
  conditions {
    client_app_types = ["all"]
    
    applications {
      included_applications = ["All"]
    }
    
    users {
      included_groups = var.create_security_groups ? [
        for group in var.security_groups : azuread_group.main[group.name].id 
        if group.name == "${var.name_prefix}-admins"
      ] : []
    }
    
    locations {
      included_locations = ["All"]
    }
  }
  
  grant_controls {
    operator          = "OR"
    built_in_controls = ["mfa"]
  }
}

# Outputs
output "admin_group_id" {
  value = var.create_security_groups && length(var.security_groups) > 0 ? (
    length([for g in var.security_groups : g if g.name == "${var.name_prefix}-admins"]) > 0 ? 
    azuread_group.main["${var.name_prefix}-admins"].id : ""
  ) : ""
}

output "developer_group_id" {
  value = var.create_security_groups && length(var.security_groups) > 0 ? (
    length([for g in var.security_groups : g if g.name == "${var.name_prefix}-developers"]) > 0 ? 
    azuread_group.main["${var.name_prefix}-developers"].id : ""
  ) : ""
}

output "operator_group_id" {
  value = var.create_security_groups && length(var.security_groups) > 0 ? (
    length([for g in var.security_groups : g if g.name == "${var.name_prefix}-operators"]) > 0 ? 
    azuread_group.main["${var.name_prefix}-operators"].id : ""
  ) : ""
}

output "security_group_ids" {
  value = var.create_security_groups ? { for k, v in azuread_group.main : k => v.id } : {}
}

output "service_principal_ids" {
  value = var.create_service_principals ? { for k, v in azuread_service_principal.main : k => v.id } : {}
}

output "service_principal_client_ids" {
  value = var.create_service_principals ? { for k, v in azuread_service_principal.main : k => v.client_id } : {}
}

output "app_registration_client_ids" {
  value = var.create_app_registrations ? { for k, v in azuread_application.app : k => v.client_id } : {}
}

output "app_registration_ids" {
  value = var.create_app_registrations ? { for k, v in azuread_application.app : k => v.id } : {}
}

output "app_registration_secrets" {
  value = var.create_app_registrations ? { 
    for k, v in azuread_application_password.app : k => {
      application_id = v.application_id
      key_id        = v.key_id
      # Secret value is not exposed in output for security
    }
  } : {}
  sensitive = true
}

output "aks_identity_id" {
  value = var.create_service_principals && contains([for sp in var.service_principals : sp.name], "${var.name_prefix}-aks-sp") ? (
    azuread_service_principal.main["${var.name_prefix}-aks-sp"].id
  ) : ""
}