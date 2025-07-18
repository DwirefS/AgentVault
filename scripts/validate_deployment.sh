#!/bin/bash
# AgentVault™ Deployment Validation Script
# Comprehensive validation of all infrastructure components

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
RESOURCE_GROUP="${RESOURCE_GROUP:-agentvault-prod-rg}"
CLUSTER_NAME="${CLUSTER_NAME:-agentvault-prod-aks}"
NAMESPACE="${NAMESPACE:-agentvault}"

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

# Header
echo "========================================="
echo "AgentVault™ Deployment Validation"
echo "========================================="
echo "Resource Group: $RESOURCE_GROUP"
echo "Cluster Name: $CLUSTER_NAME"
echo "Namespace: $NAMESPACE"
echo "========================================="
echo ""

# 1. Azure Resources Validation
log_info "Validating Azure Resources..."

# Check Resource Group
if az group show --name "$RESOURCE_GROUP" &>/dev/null; then
    check_pass "Resource Group exists"
else
    check_fail "Resource Group not found"
fi

# Check Virtual Network
if az network vnet list --resource-group "$RESOURCE_GROUP" --query "[?contains(name, 'agentvault')]" -o tsv | grep -q .; then
    check_pass "Virtual Network configured"
else
    check_fail "Virtual Network not found"
fi

# Check AKS Cluster
if az aks show --name "$CLUSTER_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    check_pass "AKS Cluster deployed"
    
    # Check node pools
    NODE_POOLS=$(az aks nodepool list --cluster-name "$CLUSTER_NAME" --resource-group "$RESOURCE_GROUP" --query "[].name" -o tsv)
    if [[ -n "$NODE_POOLS" ]]; then
        check_pass "Node pools configured: $NODE_POOLS"
    else
        check_fail "No node pools found"
    fi
else
    check_fail "AKS Cluster not found"
fi

# Check Key Vault
if az keyvault list --resource-group "$RESOURCE_GROUP" --query "[?contains(name, 'agentvault')]" -o tsv | grep -q .; then
    check_pass "Key Vault deployed"
else
    check_fail "Key Vault not found"
fi

# Check NetApp Account
if az netappfiles account list --resource-group "$RESOURCE_GROUP" --query "[?contains(name, 'agentvault')]" -o tsv | grep -q .; then
    check_pass "NetApp Account configured"
    
    # Check capacity pools
    ANF_ACCOUNT=$(az netappfiles account list --resource-group "$RESOURCE_GROUP" --query "[?contains(name, 'agentvault')].name" -o tsv | head -1)
    if [[ -n "$ANF_ACCOUNT" ]]; then
        POOLS=$(az netappfiles pool list --account-name "$ANF_ACCOUNT" --resource-group "$RESOURCE_GROUP" --query "[].name" -o tsv)
        if [[ -n "$POOLS" ]]; then
            check_pass "ANF Capacity pools: $POOLS"
        else
            check_fail "No ANF capacity pools found"
        fi
    fi
else
    check_fail "NetApp Account not found"
fi

# Check PostgreSQL
if az postgres flexible-server list --resource-group "$RESOURCE_GROUP" --query "[?contains(name, 'agentvault')]" -o tsv | grep -q .; then
    check_pass "PostgreSQL Flexible Server deployed"
else
    check_fail "PostgreSQL server not found"
fi

# Check Redis Cache
if az redis list --resource-group "$RESOURCE_GROUP" --query "[?contains(name, 'agentvault')]" -o tsv | grep -q .; then
    check_pass "Redis Cache deployed"
else
    check_fail "Redis Cache not found"
fi

# Check Container Registry
if az acr list --resource-group "$RESOURCE_GROUP" --query "[?contains(name, 'agentvault')]" -o tsv | grep -q .; then
    check_pass "Container Registry deployed"
else
    check_fail "Container Registry not found"
fi

# Check Log Analytics
if az monitor log-analytics workspace list --resource-group "$RESOURCE_GROUP" --query "[?contains(name, 'agentvault')]" -o tsv | grep -q .; then
    check_pass "Log Analytics Workspace configured"
else
    check_fail "Log Analytics Workspace not found"
fi

# Check Application Insights
if az monitor app-insights component list --resource-group "$RESOURCE_GROUP" --query "[?contains(name, 'agentvault')]" -o tsv | grep -q .; then
    check_pass "Application Insights configured"
else
    check_fail "Application Insights not found"
fi

# Check Recovery Services Vault
if az backup vault list --resource-group "$RESOURCE_GROUP" --query "[?contains(name, 'agentvault')]" -o tsv | grep -q .; then
    check_pass "Recovery Services Vault configured"
else
    check_fail "Recovery Services Vault not found"
fi

echo ""

# 2. Kubernetes Resources Validation
log_info "Validating Kubernetes Resources..."

# Get AKS credentials
if az aks get-credentials --name "$CLUSTER_NAME" --resource-group "$RESOURCE_GROUP" --overwrite-existing &>/dev/null; then
    check_pass "AKS credentials retrieved"
else
    check_fail "Failed to get AKS credentials"
    log_error "Cannot proceed with Kubernetes validation"
    exit 1
fi

# Check cluster connection
if kubectl cluster-info &>/dev/null; then
    check_pass "Connected to Kubernetes cluster"
else
    check_fail "Cannot connect to Kubernetes cluster"
fi

# Check namespaces
for ns in agentvault monitoring ingress-nginx cert-manager; do
    if kubectl get namespace "$ns" &>/dev/null; then
        check_pass "Namespace '$ns' exists"
    else
        check_fail "Namespace '$ns' not found"
    fi
done

# Check deployments in agentvault namespace
DEPLOYMENTS=$(kubectl get deployments -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
if [[ $DEPLOYMENTS -gt 0 ]]; then
    check_pass "Found $DEPLOYMENTS deployments in $NAMESPACE namespace"
    
    # Check deployment status
    READY_DEPLOYMENTS=$(kubectl get deployments -n "$NAMESPACE" --no-headers 2>/dev/null | grep -c "1/1" || true)
    if [[ $READY_DEPLOYMENTS -eq $DEPLOYMENTS ]]; then
        check_pass "All deployments are ready"
    else
        check_fail "Some deployments are not ready ($READY_DEPLOYMENTS/$DEPLOYMENTS)"
    fi
else
    check_fail "No deployments found in $NAMESPACE namespace"
fi

# Check services
SERVICES=$(kubectl get services -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
if [[ $SERVICES -gt 0 ]]; then
    check_pass "Found $SERVICES services in $NAMESPACE namespace"
else
    check_fail "No services found in $NAMESPACE namespace"
fi

# Check persistent volumes
PVS=$(kubectl get pv --no-headers 2>/dev/null | grep -c "$NAMESPACE" || true)
if [[ $PVS -gt 0 ]]; then
    check_pass "Found $PVS persistent volumes"
else
    check_fail "No persistent volumes found"
fi

# Check ingress
if kubectl get ingress -n "$NAMESPACE" &>/dev/null; then
    check_pass "Ingress configured"
else
    check_fail "No ingress found"
fi

# Check horizontal pod autoscalers
HPAS=$(kubectl get hpa -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
if [[ $HPAS -gt 0 ]]; then
    check_pass "Found $HPAS horizontal pod autoscalers"
else
    check_fail "No horizontal pod autoscalers found"
fi

# Check network policies
NETPOLS=$(kubectl get networkpolicy -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
if [[ $NETPOLS -gt 0 ]]; then
    check_pass "Found $NETPOLS network policies"
else
    check_fail "No network policies found"
fi

echo ""

# 3. Security Validation
log_info "Validating Security Configuration..."

# Check RBAC
if kubectl get clusterrolebindings | grep -q azure-ad; then
    check_pass "Azure AD RBAC integration enabled"
else
    check_fail "Azure AD RBAC not configured"
fi

# Check pod security policies (or pod security standards)
if kubectl get psp &>/dev/null || kubectl get ns --show-labels | grep -q "pod-security"; then
    check_pass "Pod security configured"
else
    check_warn "Pod security policies not found (may be using Pod Security Standards)"
fi

# Check secrets
SECRETS=$(kubectl get secrets -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
if [[ $SECRETS -gt 0 ]]; then
    check_pass "Found $SECRETS secrets in namespace"
else
    check_fail "No secrets found in namespace"
fi

# Check service accounts
SAS=$(kubectl get serviceaccounts -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
if [[ $SAS -gt 1 ]]; then  # More than default SA
    check_pass "Found $SAS service accounts"
else
    check_fail "No custom service accounts found"
fi

echo ""

# 4. Monitoring Validation
log_info "Validating Monitoring Configuration..."

# Check Prometheus
if kubectl get deployment -n monitoring | grep -q prometheus; then
    check_pass "Prometheus deployed"
else
    check_fail "Prometheus not found"
fi

# Check Grafana
if kubectl get deployment -n monitoring | grep -q grafana; then
    check_pass "Grafana deployed"
else
    check_fail "Grafana not found"
fi

# Check metrics server
if kubectl get deployment metrics-server -n kube-system &>/dev/null; then
    check_pass "Metrics server deployed"
else
    check_fail "Metrics server not found"
fi

echo ""

# 5. High Availability Validation
log_info "Validating High Availability Configuration..."

# Check multiple replicas
MULTI_REPLICA_DEPLOYMENTS=$(kubectl get deployments -A --no-headers | awk '$3 > 1' | wc -l)
if [[ $MULTI_REPLICA_DEPLOYMENTS -gt 0 ]]; then
    check_pass "Found $MULTI_REPLICA_DEPLOYMENTS deployments with multiple replicas"
else
    check_fail "No deployments with multiple replicas found"
fi

# Check PodDisruptionBudgets
PDBS=$(kubectl get pdb -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
if [[ $PDBS -gt 0 ]]; then
    check_pass "Found $PDBS PodDisruptionBudgets"
else
    check_fail "No PodDisruptionBudgets found"
fi

# Check node distribution
NODES=$(kubectl get nodes --no-headers | wc -l)
if [[ $NODES -gt 2 ]]; then
    check_pass "Cluster has $NODES nodes for HA"
else
    check_fail "Insufficient nodes for HA (found $NODES)"
fi

echo ""

# 6. Backup Validation
log_info "Validating Backup Configuration..."

# Check backup vault
VAULT_NAME=$(az backup vault list --resource-group "$RESOURCE_GROUP" --query "[0].name" -o tsv)
if [[ -n "$VAULT_NAME" ]]; then
    check_pass "Backup vault configured: $VAULT_NAME"
    
    # Check backup policies
    POLICIES=$(az backup policy list --resource-group "$RESOURCE_GROUP" --vault-name "$VAULT_NAME" --query "length(@)" -o tsv)
    if [[ $POLICIES -gt 0 ]]; then
        check_pass "Found $POLICIES backup policies"
    else
        check_fail "No backup policies found"
    fi
else
    check_fail "No backup vault found"
fi

echo ""

# 7. Application Health Check
log_info "Performing Application Health Checks..."

# Get service endpoint
SERVICE_IP=$(kubectl get svc -n "$NAMESPACE" agentvault -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")

if [[ -n "$SERVICE_IP" ]]; then
    check_pass "Service has external IP: $SERVICE_IP"
    
    # Check health endpoint
    if curl -s -o /dev/null -w "%{http_code}" "http://$SERVICE_IP:8080/health" | grep -q "200"; then
        check_pass "Health endpoint responding"
    else
        check_fail "Health endpoint not responding"
    fi
else
    check_fail "Service does not have external IP"
fi

echo ""

# Summary
echo "========================================="
echo "Validation Summary"
echo "========================================="
echo -e "Total Checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
echo ""

if [[ $FAILED_CHECKS -eq 0 ]]; then
    echo -e "${GREEN}✓ All validation checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some validation checks failed. Please review the output above.${NC}"
    exit 1
fi