# AgentVault™ - Comprehensive Terraform Deployment Guide

## 🏗️ Complete Infrastructure Deployment

This guide provides step-by-step instructions for deploying the complete AgentVault™ infrastructure with all enterprise features including HA, DR, security, and monitoring.

## 📋 Prerequisites

### Required Tools
```bash
# Azure CLI
az --version  # >= 2.50.0

# Terraform
terraform --version  # >= 1.3.0

# kubectl
kubectl version --client  # >= 1.28.0

# Helm
helm version  # >= 3.12.0
```

### Azure Permissions
- Subscription Owner or Contributor role
- Azure AD Global Administrator or Application Administrator
- Key Vault Administrator
- User Access Administrator

### Required Azure Resource Providers
```bash
# Register required providers
az provider register --namespace Microsoft.NetApp
az provider register --namespace Microsoft.ContainerService
az provider register --namespace Microsoft.KeyVault
az provider register --namespace Microsoft.DBforPostgreSQL
az provider register --namespace Microsoft.Cache
az provider register --namespace Microsoft.RecoveryServices
az provider register --namespace Microsoft.Network
az provider register --namespace Microsoft.Monitor
az provider register --namespace Microsoft.Insights
az provider register --namespace Microsoft.AlertsManagement
az provider register --namespace Microsoft.Dashboard
```

## 🚀 Deployment Steps

### 1. Clone Repository
```bash
git clone https://github.com/DwirefS/AgentVault.git
cd AgentVault/terraform
```

### 2. Configure Azure Authentication
```bash
# Login to Azure
az login

# Set subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Create service principal for Terraform
az ad sp create-for-rbac \
  --name "agentvault-terraform-sp" \
  --role="Owner" \
  --scopes="/subscriptions/YOUR_SUBSCRIPTION_ID"
```

### 3. Create Terraform Backend Storage
```bash
# Create resource group for Terraform state
az group create \
  --name agentvault-terraform-state \
  --location eastus

# Create storage account
az storage account create \
  --name agentvaultterraform \
  --resource-group agentvault-terraform-state \
  --location eastus \
  --sku Standard_GRS \
  --encryption-services blob

# Create container
az storage container create \
  --name tfstate \
  --account-name agentvaultterraform \
  --auth-mode login
```

### 4. Configure Terraform Variables

Create `terraform.tfvars`:
```hcl
# Core Configuration
project_name = "agentvault"
environment  = "prod"
location     = "eastus"

# Business Configuration
cost_center         = "IT-001"
business_unit       = "AI-Platform"
monthly_budget      = 25000
budget_alert_emails = ["alerts@yourcompany.com"]

# Networking
vnet_address_space = ["10.0.0.0/16"]
allowed_ips        = ["YOUR_PUBLIC_IP/32"]

# Security
admin_object_ids = ["YOUR_AAD_OBJECT_ID"]
domain_name      = "agentvault.yourcompany.com"

# AKS Configuration
kubernetes_version      = "1.28.3"
aks_system_node_size   = "Standard_D4s_v5"
aks_agent_node_count   = 3
aks_agent_max_count    = 10
aks_agent_node_size    = "Standard_NC6s_v3"

# ANF Configuration
anf_premium_pool_size     = 4
anf_standard_pool_size    = 8
anf_ultra_pool_size       = 2
anf_agents_volume_size    = 2
anf_models_volume_size    = 4
anf_artifacts_volume_size = 2

# Database
database_storage_mb      = 512000  # 500GB
database_admin_username  = "agentvault_admin"

# Monitoring
alert_email_addresses = ["ops@yourcompany.com"]
alert_sms_numbers     = ["+1234567890"]
alert_webhooks        = ["https://webhook.site/your-webhook"]

# Disaster Recovery
enable_disaster_recovery = true

# Feature Flags
enable_autoscaling   = true
enable_monitoring    = true
enable_backup        = true
enable_encryption    = true
```

### 5. Initialize Terraform
```bash
# Initialize with backend config
terraform init \
  -backend-config="resource_group_name=agentvault-terraform-state" \
  -backend-config="storage_account_name=agentvaultterraform" \
  -backend-config="container_name=tfstate" \
  -backend-config="key=agentvault.terraform.tfstate"
```

### 6. Plan Deployment
```bash
# Review the deployment plan
terraform plan -out=agentvault.tfplan

# Save plan summary
terraform show -json agentvault.tfplan > plan-summary.json
```

### 7. Deploy Infrastructure
```bash
# Deploy in phases for better control
# Phase 1: Core infrastructure
terraform apply -target=module.networking -target=module.security -target=module.identity

# Phase 2: Storage and Database
terraform apply -target=module.storage -target=module.database -target=module.redis

# Phase 3: Compute and Monitoring
terraform apply -target=module.aks -target=module.monitoring

# Phase 4: Backup and DR
terraform apply -target=module.backup -target=module.disaster_recovery

# Phase 5: Complete deployment
terraform apply agentvault.tfplan
```

### 8. Configure Kubernetes
```bash
# Get AKS credentials
az aks get-credentials \
  --resource-group agentvault-prod-rg \
  --name agentvault-prod-aks

# Verify connection
kubectl get nodes

# Create namespaces
kubectl create namespace agentvault
kubectl create namespace monitoring
kubectl create namespace ingress-nginx
```

### 9. Install Kubernetes Components
```bash
# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/azure-load-balancer-health-probe-request-path"=/healthz

# Install cert-manager
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true

# Install Prometheus and Grafana (if not using Azure Monitor)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring
```

### 10. Deploy AgentVault Application
```bash
# Apply Kubernetes manifests
kubectl apply -f ../deployment/k8s/production/

# Verify deployment
kubectl get all -n agentvault

# Check pod logs
kubectl logs -n agentvault -l app=agentvault -f
```

## 🔧 Post-Deployment Configuration

### 1. Configure DNS
```bash
# Get ingress IP
INGRESS_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Configure DNS A record
az network dns record-set a add-record \
  --resource-group YOUR_DNS_RG \
  --zone-name yourcompany.com \
  --record-set-name agentvault \
  --ipv4-address $INGRESS_IP
```

### 2. Configure SSL/TLS
```yaml
# Create certificate issuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourcompany.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### 3. Configure Monitoring Alerts
```bash
# Apply alert rules
terraform output -json monitoring_endpoints | jq -r
```

### 4. Test Disaster Recovery
```bash
# Run DR health check
az automation runbook start \
  --resource-group agentvault-prod-dr-rg \
  --automation-account-name agentvault-prod-dr-automation \
  --name agentvault-prod-health-check-runbook
```

## 📊 Validation Checklist

### Infrastructure Components
- [ ] Resource Groups created in primary and DR regions
- [ ] Virtual Networks with proper subnets
- [ ] Network Security Groups with rules applied
- [ ] Azure Firewall and Bastion (if enabled)
- [ ] Private DNS zones created and linked

### Security
- [ ] Key Vault with encryption keys
- [ ] Managed identities created
- [ ] RBAC assignments configured
- [ ] Azure AD groups and service principals
- [ ] Network policies enforced

### Compute
- [ ] AKS cluster with multiple node pools
- [ ] Container Registry configured
- [ ] Workload identity enabled
- [ ] Pod security policies applied

### Storage
- [ ] NetApp account and capacity pools
- [ ] ANF volumes mounted in AKS
- [ ] Backup storage account
- [ ] Encryption at rest enabled

### Database
- [ ] PostgreSQL Flexible Server with HA
- [ ] Database created and configured
- [ ] Connection pooling optimized
- [ ] Audit logging enabled

### Monitoring
- [ ] Log Analytics workspace
- [ ] Application Insights
- [ ] Prometheus and Grafana
- [ ] Alert rules and action groups
- [ ] Dashboards and workbooks

### Backup & DR
- [ ] Recovery Services Vault
- [ ] Backup policies configured
- [ ] Cross-region replication
- [ ] Traffic Manager for failover
- [ ] DR automation runbooks

## 🔍 Monitoring URLs

After deployment, access these endpoints:

- **Application**: https://agentvault.yourcompany.com
- **Grafana**: https://agentvault.yourcompany.com/grafana
- **Prometheus**: https://agentvault.yourcompany.com/prometheus
- **API Documentation**: https://agentvault.yourcompany.com/docs

## 🚨 Troubleshooting

### Common Issues

1. **Terraform State Lock**
```bash
terraform force-unlock LOCK_ID
```

2. **AKS Node Issues**
```bash
kubectl describe node NODE_NAME
az aks nodepool upgrade --cluster-name agentvault-prod-aks --name agents --resource-group agentvault-prod-rg
```

3. **ANF Mount Issues**
```bash
kubectl describe pv
kubectl logs -n agentvault deployment/agentvault
```

4. **Database Connection**
```bash
# Test connection
kubectl run -it --rm psql --image=postgres:15 --restart=Never -- psql -h DATABASE_FQDN -U agentvault_admin -d agentvault
```

## 💰 Cost Optimization

### Estimated Monthly Costs (Production)
- AKS Cluster: ~$800-1200
- ANF Storage: ~$2000-3000
- PostgreSQL HA: ~$500-800
- Application Gateway: ~$200-300
- Monitoring: ~$300-500
- Backup & DR: ~$500-800
- **Total**: ~$4,300-6,600/month

### Cost Saving Tips
1. Use spot instances for non-critical workloads
2. Enable auto-shutdown for dev/test environments
3. Right-size resources based on actual usage
4. Use reserved instances for stable workloads
5. Configure lifecycle policies for storage

## 🔐 Security Best Practices

1. **Enable MFA** for all administrative accounts
2. **Rotate secrets** regularly using Key Vault
3. **Review RBAC** assignments monthly
4. **Monitor security alerts** in Azure Security Center
5. **Conduct DR drills** quarterly
6. **Update dependencies** monthly
7. **Review audit logs** weekly

## 📝 Maintenance Tasks

### Daily
- Monitor application health
- Check backup status
- Review security alerts

### Weekly
- Review performance metrics
- Check cost trends
- Update documentation

### Monthly
- Rotate credentials
- Update dependencies
- Performance optimization
- Security patches

### Quarterly
- DR drill
- Capacity planning
- Architecture review
- Cost optimization

## 🆘 Support

For issues or questions:
- **Email**: DwirefS@SapientEdge.io
- **Documentation**: [GitHub Wiki](https://github.com/DwirefS/AgentVault/wiki)
- **Issues**: [GitHub Issues](https://github.com/DwirefS/AgentVault/issues)

---

**Last Updated**: $(date)
**Version**: 1.0.0
**Author**: Dwiref Sharma