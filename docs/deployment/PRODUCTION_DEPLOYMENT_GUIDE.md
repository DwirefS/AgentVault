# AgentVault™ Production Deployment Guide

## 📋 Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Infrastructure Deployment](#infrastructure-deployment)
3. [Application Deployment](#application-deployment)
4. [Security Configuration](#security-configuration)
5. [Monitoring Setup](#monitoring-setup)
6. [Performance Tuning](#performance-tuning)
7. [Disaster Recovery](#disaster-recovery)
8. [Operational Procedures](#operational-procedures)
9. [Troubleshooting](#troubleshooting)

## 🔍 Pre-Deployment Checklist

### Azure Prerequisites
- [ ] Azure subscription with sufficient quota:
  - Compute: 100+ vCPUs
  - Storage: 50TB minimum
  - Network: 10+ public IPs
- [ ] Azure AD tenant configured
- [ ] Required resource providers registered
- [ ] Service Principal with Owner role
- [ ] DNS zone configured

### Security Requirements
- [ ] SSL certificates procured
- [ ] Firewall rules documented
- [ ] Security team approval
- [ ] Compliance requirements verified
- [ ] Penetration testing completed

### Team Readiness
- [ ] Operations team trained
- [ ] Runbooks prepared
- [ ] On-call rotation established
- [ ] Escalation procedures defined
- [ ] Communication channels setup

## 🏗️ Infrastructure Deployment

### 1. Initial Setup

```bash
# Clone the repository
git clone https://github.com/DwirefS/AgentVault.git
cd AgentVault

# Setup environment
export ENVIRONMENT="production"
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_LOCATION="eastus2"
```

### 2. Terraform Deployment

```bash
# Navigate to Terraform directory
cd terraform

# Initialize Terraform with backend
terraform init \
  -backend-config="resource_group_name=agentvault-terraform-state" \
  -backend-config="storage_account_name=agentvaultterraform" \
  -backend-config="container_name=tfstate" \
  -backend-config="key=prod.terraform.tfstate"

# Create production workspace
terraform workspace new production || terraform workspace select production

# Review the plan
terraform plan -var-file="environments/prod/terraform.tfvars" -out=prod.tfplan

# Apply infrastructure (this will take 30-45 minutes)
terraform apply prod.tfplan
```

### 3. Post-Infrastructure Validation

```bash
# Run infrastructure validation
./scripts/validate_infrastructure.sh

# Expected output:
# ✓ Resource Groups created
# ✓ Virtual Network configured
# ✓ AKS cluster deployed
# ✓ Azure NetApp Files provisioned
# ✓ Key Vault accessible
# ✓ Monitoring enabled
```

## 🚀 Application Deployment

### 1. AKS Configuration

```bash
# Get AKS credentials
az aks get-credentials \
  --resource-group agentvault-prod-rg \
  --name agentvault-prod-aks \
  --admin

# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

### 2. Namespace Setup

```bash
# Create namespaces
kubectl create namespace agentvault
kubectl create namespace monitoring
kubectl create namespace ingress-nginx

# Label namespaces
kubectl label namespace agentvault environment=production
kubectl label namespace agentvault app=agentvault
```

### 3. Secret Management

```bash
# Create Azure AD secret
kubectl create secret generic azure-ad-secret \
  --namespace agentvault \
  --from-literal=client-id=$AZURE_CLIENT_ID \
  --from-literal=client-secret=$AZURE_CLIENT_SECRET \
  --from-literal=tenant-id=$AZURE_TENANT_ID

# Create database credentials
kubectl create secret generic database-credentials \
  --namespace agentvault \
  --from-literal=username=agentvault_admin \
  --from-literal=password=$(terraform output -raw database_admin_password)

# Create Redis credentials
kubectl create secret generic redis-credentials \
  --namespace agentvault \
  --from-literal=connection-string=$(terraform output -raw redis_connection_string)
```

### 4. Deploy Core Components

```bash
# Deploy ConfigMaps
kubectl apply -f deployment/k8s/production/configmaps/

# Deploy RBAC
kubectl apply -f deployment/k8s/production/rbac/

# Deploy Storage Classes
kubectl apply -f deployment/k8s/production/storage/

# Deploy Network Policies
kubectl apply -f deployment/k8s/production/network-policies/
```

### 5. Deploy AgentVault Application

```bash
# Deploy main application
kubectl apply -f deployment/k8s/production/agentvault-deployment.yaml

# Wait for rollout
kubectl rollout status deployment/agentvault -n agentvault

# Verify pods are running
kubectl get pods -n agentvault -l app=agentvault
```

### 6. Deploy Supporting Services

```bash
# Deploy Redis cluster
helm install redis bitnami/redis-cluster \
  --namespace agentvault \
  --values deployment/helm/values/redis-production.yaml

# Deploy monitoring stack
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values deployment/helm/values/monitoring-production.yaml

# Deploy ingress controller
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --values deployment/helm/values/ingress-production.yaml
```

## 🔒 Security Configuration

### 1. Network Security

```bash
# Apply network policies
kubectl apply -f deployment/k8s/production/network-policies/

# Configure WAF rules
az network application-gateway waf-policy rule create \
  --policy-name agentvault-waf-policy \
  --resource-group agentvault-prod-rg \
  --name BlockSQLInjection \
  --priority 100 \
  --rule-type MatchRule \
  --action Block
```

### 2. Pod Security

```yaml
# Apply Pod Security Standards
kubectl label namespace agentvault pod-security.kubernetes.io/enforce=restricted
kubectl label namespace agentvault pod-security.kubernetes.io/audit=restricted
kubectl label namespace agentvault pod-security.kubernetes.io/warn=restricted
```

### 3. RBAC Configuration

```bash
# Create service accounts
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: agentvault-workload-identity
  namespace: agentvault
  annotations:
    azure.workload.identity/client-id: $(terraform output -raw aks_workload_identity_client_id)
EOF

# Bind roles
kubectl create clusterrolebinding agentvault-view \
  --clusterrole=view \
  --serviceaccount=agentvault:agentvault-workload-identity
```

### 4. SSL/TLS Configuration

```bash
# Create certificate
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: agentvault-tls
  namespace: agentvault
spec:
  secretName: agentvault-tls-secret
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - agentvault.yourdomain.com
  - api.agentvault.yourdomain.com
EOF
```

## 📊 Monitoring Setup

### 1. Configure Prometheus

```bash
# Apply Prometheus configuration
kubectl apply -f deployment/k8s/production/monitoring/prometheus-config.yaml

# Create ServiceMonitor for AgentVault
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: agentvault
  namespace: agentvault
spec:
  selector:
    matchLabels:
      app: agentvault
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
EOF
```

### 2. Configure Grafana Dashboards

```bash
# Import AgentVault dashboards
kubectl create configmap agentvault-dashboards \
  --namespace monitoring \
  --from-file=deployment/grafana/dashboards/

# Configure datasources
kubectl apply -f deployment/k8s/production/monitoring/grafana-datasources.yaml
```

### 3. Setup Alerts

```bash
# Apply alert rules
kubectl apply -f deployment/k8s/production/monitoring/alerts/

# Configure alert channels
kubectl create secret generic alertmanager-slack \
  --namespace monitoring \
  --from-literal=webhook-url=$SLACK_WEBHOOK_URL
```

### 4. Enable Application Insights

```bash
# Configure Application Insights
export INSTRUMENTATION_KEY=$(terraform output -raw application_insights_instrumentation_key)

kubectl create configmap appinsights-config \
  --namespace agentvault \
  --from-literal=instrumentation-key=$INSTRUMENTATION_KEY
```

## ⚡ Performance Tuning

### 1. AKS Node Pool Optimization

```bash
# Scale agent node pool
az aks nodepool scale \
  --resource-group agentvault-prod-rg \
  --cluster-name agentvault-prod-aks \
  --name agents \
  --node-count 5

# Enable cluster autoscaler
az aks nodepool update \
  --resource-group agentvault-prod-rg \
  --cluster-name agentvault-prod-aks \
  --name agents \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10
```

### 2. Application Performance

```yaml
# Update resource limits
kubectl patch deployment agentvault -n agentvault --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/limits/cpu",
    "value": "4000m"
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/limits/memory",
    "value": "8Gi"
  }
]'
```

### 3. Storage Performance

```bash
# Optimize ANF volumes
az netappfiles volume update \
  --resource-group agentvault-prod-rg \
  --account-name agentvault-prod-anf \
  --pool-name premium \
  --name agents \
  --throughput-mibps 256
```

### 4. Cache Optimization

```bash
# Scale Redis cluster
helm upgrade redis bitnami/redis-cluster \
  --namespace agentvault \
  --set cluster.nodes=6 \
  --set cluster.replicas=1
```

## 🔄 Disaster Recovery

### 1. Enable Cross-Region Replication

```bash
# Setup DR region
export DR_LOCATION="westus2"

# Deploy DR infrastructure
cd terraform
terraform workspace new dr
terraform apply -var-file="environments/dr/terraform.tfvars"
```

### 2. Configure Database Replication

```bash
# Enable geo-replication for PostgreSQL
az postgres flexible-server replica create \
  --resource-group agentvault-dr-rg \
  --name agentvault-dr-postgres \
  --source-server $(terraform output -raw database_server_id)
```

### 3. ANF Cross-Region Replication

```bash
# Create replication for each volume
for volume in agents models artifacts; do
  az netappfiles volume replication create \
    --resource-group agentvault-prod-rg \
    --account-name agentvault-prod-anf \
    --pool-name premium \
    --volume-name $volume \
    --endpoint-type dst \
    --remote-volume-resource-id "/subscriptions/.../volumes/$volume-dr" \
    --replication-schedule hourly
done
```

### 4. Test Failover Procedures

```bash
# Run DR drill
./scripts/dr-drill.sh \
  --source-region $AZURE_LOCATION \
  --target-region $DR_LOCATION \
  --dry-run

# Validate DR readiness
./scripts/validate-dr.sh
```

## 📋 Operational Procedures

### Daily Operations

1. **Health Checks**
   ```bash
   # Check cluster health
   kubectl get nodes
   kubectl get pods -A | grep -v Running
   
   # Check application health
   curl https://agentvault.yourdomain.com/health
   ```

2. **Backup Verification**
   ```bash
   # List recent backups
   az backup item list \
     --resource-group agentvault-backup-rg \
     --vault-name agentvault-prod-rsv
   ```

3. **Performance Monitoring**
   ```bash
   # Check metrics
   kubectl top nodes
   kubectl top pods -n agentvault
   ```

### Weekly Operations

1. **Security Patching**
   ```bash
   # Update AKS nodes
   az aks nodepool upgrade \
     --resource-group agentvault-prod-rg \
     --cluster-name agentvault-prod-aks \
     --name system \
     --node-image-only
   ```

2. **Capacity Planning**
   ```bash
   # Review usage trends
   ./scripts/capacity-report.sh --days 7
   ```

### Monthly Operations

1. **Disaster Recovery Drill**
   ```bash
   # Execute monthly DR test
   ./scripts/dr-drill.sh --monthly-test
   ```

2. **Security Audit**
   ```bash
   # Run security scan
   ./scripts/security-audit.sh --full-scan
   ```

## 🔧 Troubleshooting

### Common Issues

#### 1. Pod Crashes
```bash
# Check pod logs
kubectl logs -n agentvault <pod-name> --previous

# Describe pod for events
kubectl describe pod -n agentvault <pod-name>

# Check resource constraints
kubectl top pod -n agentvault <pod-name>
```

#### 2. Storage Issues
```bash
# Check PVC status
kubectl get pvc -n agentvault

# Verify ANF volumes
az netappfiles volume list \
  --resource-group agentvault-prod-rg \
  --account-name agentvault-prod-anf \
  --pool-name premium
```

#### 3. Authentication Failures
```bash
# Check Azure AD integration
kubectl logs -n agentvault deployment/agentvault | grep -i auth

# Verify service principal
az ad sp show --id $AZURE_CLIENT_ID
```

#### 4. Performance Degradation
```bash
# Check metrics
curl -s http://localhost:8080/metrics | grep agentvault_request_duration

# Profile application
kubectl exec -n agentvault <pod-name> -- python -m cProfile -o profile.out main.py
```

### Emergency Procedures

#### Complete System Failure
1. Activate incident response team
2. Check Azure Service Health
3. Initiate DR failover if needed
4. Notify stakeholders

#### Data Corruption
1. Stop write operations
2. Identify corruption scope
3. Restore from backup
4. Validate data integrity

#### Security Breach
1. Isolate affected systems
2. Revoke compromised credentials
3. Enable emergency security policies
4. Begin forensic analysis

## 📞 Support Escalation

### Level 1: Operations Team
- Available: 24/7
- Scope: Basic troubleshooting, restarts
- Contact: ops@yourcompany.com

### Level 2: Platform Team
- Available: Business hours + on-call
- Scope: Infrastructure issues, performance
- Contact: platform@yourcompany.com

### Level 3: Engineering Team
- Available: On-call rotation
- Scope: Code issues, critical bugs
- Contact: engineering@yourcompany.com

### Vendor Support
- **Microsoft Azure**: Premium support contract
- **AgentVault™**: DwirefS@SapientEdge.io

## 📚 Additional Resources

- [Terraform Deployment Guide](../../TERRAFORM_DEPLOYMENT_GUIDE.md)
- [API Documentation](../api/openapi.yaml)
- [Architecture Diagrams](../architecture/)
- [Runbooks](../operations/runbooks/)
- [Security Policies](../security/)

---

**Document Version**: 1.0.0  
**Last Updated**: $(date)  
**Maintained By**: AgentVault Platform Team