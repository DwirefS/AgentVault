# AgentVault™ Complete Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Deployment Options](#deployment-options)
4. [Step-by-Step Deployment](#step-by-step-deployment)
5. [Configuration](#configuration)
6. [Validation & Testing](#validation--testing)
7. [Production Checklist](#production-checklist)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools
- **Azure CLI** >= 2.56.0
- **Terraform** >= 1.0
- **Python** >= 3.9
- **Docker** >= 20.10
- **kubectl** >= 1.25 (for Kubernetes deployments)
- **Git** >= 2.30

### Azure Requirements
- Active Azure subscription
- Sufficient quota for:
  - Azure NetApp Files (minimum 4TB)
  - Virtual Networks
  - Compute instances
  - Redis Cache (Premium tier)
  - Key Vault (Premium tier)

### Permissions Required
```bash
# Check your current permissions
az account show

# Required roles
- Contributor (subscription or resource group level)
- Network Contributor (for VNet operations)
- Key Vault Contributor
- Storage Account Contributor
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Internet Gateway                          │
└────────────────────────────┬───────────────────────────────────┘
                             │
┌────────────────────────────┴───────────────────────────────────┐
│                    Azure Application Gateway                     │
│                         (WAF Enabled)                           │
└────────────────────────────┬───────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────────┐
│                    Virtual Network (10.0.0.0/16)                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │   ANF Subnet        │  │  Compute Subnet  │  │  Bastion   │ │
│  │   10.0.1.0/24      │  │  10.0.2.0/24     │  │ 10.0.3.0/24│ │
│  │                    │  │                  │  │            │ │
│  │ ┌────────────────┐ │  │ ┌──────────────┐│  │            │ │
│  │ │ Ultra Volume   │ │  │ │ Orchestrator ││  │            │ │
│  │ │ Premium Volume │ │  │ │ API Services ││  │            │ │
│  │ │ Standard Vol. │ │  │ │ ML Services  ││  │            │ │
│  │ └────────────────┘ │  │ └──────────────┘│  │            │ │
│  └─────────────────────┘  └──────────────────┘  └────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               Private Endpoints Subnet                   │   │
│  │                    10.0.4.0/24                          │   │
│  │  • Key Vault  • Storage  • Redis  • Container Registry  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┴───────────────────────────────────┐
│                    Azure Backbone Network                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  ┌────────┐ │
│  │ Key Vault   │  │Redis Cache  │  │ App        │  │ Azure  │ │
│  │ (Premium)   │  │ (Premium)   │  │ Insights   │  │ AD     │ │
│  └─────────────┘  └─────────────┘  └────────────┘  └────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Deployment Options

### Option 1: Automated Deployment (Recommended)
```bash
# One-command deployment
./scripts/deploy.sh --environment prod --location "East US 2"
```

### Option 2: Terraform Deployment
```bash
cd terraform
terraform init
terraform plan -var-file="environments/prod/terraform.tfvars"
terraform apply -var-file="environments/prod/terraform.tfvars"
```

### Option 3: Kubernetes Deployment
```bash
# Deploy to AKS
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

### Option 4: Container Apps Deployment
```bash
# Deploy to Azure Container Apps
az containerapp env create \
  --name agentvault-env \
  --resource-group agentvault-prod-rg \
  --location "East US 2"

az containerapp create \
  --name agentvault-orchestrator \
  --resource-group agentvault-prod-rg \
  --environment agentvault-env \
  --image agentvault/orchestrator:latest \
  --target-port 8000 \
  --ingress 'external' \
  --min-replicas 3 \
  --max-replicas 10
```

---

## Step-by-Step Deployment

### Step 1: Clone Repository
```bash
git clone https://github.com/DwirefS/AgentVault.git
cd AgentVault
```

### Step 2: Configure Environment
```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
vim .env

# Required variables
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=agentvault-prod-rg
AZURE_LOCATION="East US 2"
ADMIN_EMAIL=your-email@company.com
```

### Step 3: Login to Azure
```bash
# Login
az login

# Set subscription
az account set --subscription $AZURE_SUBSCRIPTION_ID

# Verify
az account show
```

### Step 4: Register Providers
```bash
# Register required providers
az provider register --namespace Microsoft.NetApp --wait
az provider register --namespace Microsoft.Storage --wait
az provider register --namespace Microsoft.KeyVault --wait
az provider register --namespace Microsoft.Cache --wait
```

### Step 5: Deploy Infrastructure
```bash
# Run deployment script
./scripts/deploy.sh \
  --environment prod \
  --location "$AZURE_LOCATION" \
  --resource-group "$AZURE_RESOURCE_GROUP"

# Monitor deployment (takes ~20 minutes)
# The script will output progress
```

### Step 6: Initialize AgentVault™
```bash
# Export Terraform outputs
cd terraform
terraform output -json > ../configs/azure/terraform-output.json
cd ..

# Initialize system
agentvault init --config configs/azure/terraform-output.json

# Verify initialization
agentvault status
```

### Step 7: Deploy Sample Agent
```bash
# Deploy a test agent
python examples/test_agent.py

# Verify deployment
agentvault agents list
```

---

## Configuration

### Core Configuration (config.yaml)
```yaml
agentvault:
  environment: production
  log_level: INFO
  
azure:
  subscription_id: ${AZURE_SUBSCRIPTION_ID}
  resource_group: ${AZURE_RESOURCE_GROUP}
  location: ${AZURE_LOCATION}
  
anf:
  account_name: agentvault-prod-anf
  volumes:
    ultra:
      size_gb: 1024
      service_level: Ultra
    premium:
      size_gb: 2048
      service_level: Premium
    standard:
      size_gb: 4096
      service_level: Standard
      
redis:
  sku: Premium
  capacity: 6
  shards: 3
  
security:
  encryption_level: maximum
  key_rotation_days: 90
  enable_audit_logs: true
  
performance:
  enable_cognitive_balancing: true
  enable_neural_compression: true
  enable_predictive_caching: true
  cache_ttl_seconds: 3600
  
monitoring:
  enable_metrics: true
  enable_tracing: true
  retention_days: 90
```

### Network Security Rules
```bash
# Create NSG rules
az network nsg rule create \
  --resource-group $AZURE_RESOURCE_GROUP \
  --nsg-name agentvault-nsg \
  --name AllowHTTPS \
  --priority 100 \
  --direction Inbound \
  --access Allow \
  --protocol Tcp \
  --source-address-prefixes Internet \
  --source-port-ranges '*' \
  --destination-address-prefixes VirtualNetwork \
  --destination-port-ranges 443
```

### Storage Tier Configuration
```yaml
storage_tiers:
  ultra:
    latency_target_ms: 0.1
    iops_limit: 450000
    throughput_mbps: 4500
    
  premium:
    latency_target_ms: 1.0
    iops_limit: 64000
    throughput_mbps: 1024
    
  standard:
    latency_target_ms: 10.0
    iops_limit: 16000
    throughput_mbps: 250
```

---

## Validation & Testing

### 1. Infrastructure Validation
```bash
# Check all resources are created
az resource list --resource-group $AZURE_RESOURCE_GROUP --output table

# Verify NetApp account
az netappfiles account show \
  --resource-group $AZURE_RESOURCE_GROUP \
  --account-name agentvault-prod-anf

# Check Redis status
az redis show \
  --name agentvault-prod-redis \
  --resource-group $AZURE_RESOURCE_GROUP
```

### 2. Connectivity Testing
```bash
# Test ANF mount
sudo mkdir -p /mnt/agentvault/test
sudo mount -t nfs -o rw,hard,rsize=65536,wsize=65536,vers=4.1,tcp \
  10.0.1.4:/ultra-volume /mnt/agentvault/test

# Test Redis connectivity
redis-cli -h agentvault-prod-redis.redis.cache.windows.net \
  -p 6380 --tls --askpass ping
```

### 3. Performance Testing
```bash
# Run performance benchmarks
python tests/performance/benchmark_storage.py

# Expected results:
# Vector search: <0.1ms
# Memory retrieval: <0.2ms
# Bulk operations: <15ms
```

### 4. Integration Testing
```bash
# Run integration tests
pytest tests/integration/ -v

# Test LangChain integration
python examples/langchain/test_integration.py

# Test AutoGen integration
python examples/autogen/test_integration.py
```

---

## Production Checklist

### Pre-Deployment
- [ ] Azure subscription verified
- [ ] Resource quotas confirmed
- [ ] Network topology planned
- [ ] Security requirements documented
- [ ] Compliance requirements identified
- [ ] Backup strategy defined
- [ ] Disaster recovery plan created

### Deployment
- [ ] Infrastructure deployed via Terraform
- [ ] Network security groups configured
- [ ] Private endpoints created
- [ ] Encryption at rest enabled
- [ ] Encryption in transit configured
- [ ] Monitoring enabled
- [ ] Alerting configured

### Post-Deployment
- [ ] All services healthy
- [ ] Performance benchmarks passed
- [ ] Security scan completed
- [ ] Backup tested
- [ ] Documentation updated
- [ ] Team trained
- [ ] Runbooks created

### Security Hardening
- [ ] Multi-factor authentication enabled
- [ ] Least privilege access configured
- [ ] Network isolation verified
- [ ] Key rotation scheduled
- [ ] Audit logging enabled
- [ ] Vulnerability scanning scheduled
- [ ] Incident response plan tested

---

## Troubleshooting

### Common Issues

#### Issue: ANF Volume Mount Fails
```bash
# Check network connectivity
nslookup 10.0.1.4

# Verify subnet delegation
az network vnet subnet show \
  --resource-group $AZURE_RESOURCE_GROUP \
  --vnet-name agentvault-vnet \
  --name anf-subnet

# Solution: Ensure subnet is delegated to Microsoft.NetApp/volumes
```

#### Issue: Redis Connection Timeout
```bash
# Check firewall rules
az redis firewall-rules list \
  --name agentvault-prod-redis \
  --resource-group $AZURE_RESOURCE_GROUP

# Add your IP if needed
az redis firewall-rules create \
  --name allow-my-ip \
  --resource-group $AZURE_RESOURCE_GROUP \
  --redis-name agentvault-prod-redis \
  --start-ip YOUR_IP \
  --end-ip YOUR_IP
```

#### Issue: High Latency
```bash
# Check performance metrics
agentvault metrics --component storage

# Analyze slow queries
agentvault debug slow-queries --last 1h

# Solution: Enable cognitive load balancing
agentvault config set performance.enable_cognitive_balancing true
```

#### Issue: Authentication Failures
```bash
# Verify managed identity
az identity show \
  --name agentvault-identity \
  --resource-group $AZURE_RESOURCE_GROUP

# Check Key Vault access policies
az keyvault show \
  --name agentvault-prod-kv \
  --query "properties.accessPolicies"
```

### Debug Commands
```bash
# Get system status
agentvault status --verbose

# Check agent health
agentvault agents health --agent-id YOUR_AGENT_ID

# View logs
agentvault logs --component orchestrator --tail 100

# Performance profiling
agentvault profile --duration 60s --output profile.html
```

### Support Resources
- **Documentation**: https://agentvault.readthedocs.io
- **GitHub Issues**: https://github.com/DwirefS/AgentVault/issues
- **Email Support**: DwirefS@SapientEdge.io
- **Community Discord**: https://discord.gg/agentvault

---

## Next Steps

1. **Deploy Your First Agent**
   ```bash
   python examples/quickstart/deploy_first_agent.py
   ```

2. **Configure Monitoring**
   ```bash
   ./scripts/setup_monitoring.sh
   ```

3. **Set Up CI/CD**
   ```bash
   # GitHub Actions workflow included
   cp .github/workflows/deploy.yml.example .github/workflows/deploy.yml
   ```

4. **Schedule Backups**
   ```bash
   agentvault backup schedule \
     --frequency daily \
     --retention 30 \
     --time "02:00"
   ```

Congratulations! Your AgentVault™ deployment is complete and ready for production AI agent workloads. 🎉