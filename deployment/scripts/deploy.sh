#!/bin/bash

# AgentVault™ Deployment Script
# Production-ready deployment automation for Azure AKS
# Author: Dwiref Sharma
# Contact: DwirefS@SapientEdge.io

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
HELM_CHART_DIR="${PROJECT_ROOT}/deployment/helm/agentvault"
VALUES_DIR="${PROJECT_ROOT}/deployment/values"

# Default values
ENVIRONMENT="${ENVIRONMENT:-development}"
CLUSTER_NAME="${CLUSTER_NAME:-agentvault-aks}"
RESOURCE_GROUP="${RESOURCE_GROUP:-agentvault-rg}"
SUBSCRIPTION_ID="${SUBSCRIPTION_ID:-}"
HELM_RELEASE_NAME="${HELM_RELEASE_NAME:-agentvault}"
NAMESPACE="${NAMESPACE:-agentvault}"
REGION="${REGION:-eastus}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_INFRA="${SKIP_INFRA:-false}"
FORCE_RECREATE="${FORCE_RECREATE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Help function
show_help() {
    cat << EOF
AgentVault™ Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENVIRONMENT    Target environment (development|staging|production) [default: development]
    -c, --cluster CLUSTER_NAME       AKS cluster name [default: agentvault-aks]
    -g, --resource-group RG_NAME     Azure resource group [default: agentvault-rg]
    -s, --subscription SUBSCRIPTION  Azure subscription ID
    -r, --release RELEASE_NAME       Helm release name [default: agentvault]
    -n, --namespace NAMESPACE        Kubernetes namespace [default: agentvault]
    -l, --location REGION           Azure region [default: eastus]
    --dry-run                        Perform a dry run without making changes
    --skip-infra                     Skip infrastructure provisioning
    --force-recreate                 Force recreate existing resources
    -h, --help                       Show this help message

EXAMPLES:
    # Deploy to development environment
    $0 -e development

    # Deploy to production with specific settings
    $0 -e production -c prod-aks -g prod-rg -s your-subscription-id

    # Dry run deployment
    $0 -e staging --dry-run

    # Skip infrastructure and deploy application only
    $0 -e production --skip-infra

ENVIRONMENT VARIABLES:
    AZURE_CLIENT_ID                  Service principal client ID
    AZURE_CLIENT_SECRET              Service principal client secret
    AZURE_TENANT_ID                  Azure tenant ID
    KUBECONFIG                       Path to kubeconfig file
    HELM_DEBUG                       Enable Helm debug mode

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -c|--cluster)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        -g|--resource-group)
            RESOURCE_GROUP="$2"
            shift 2
            ;;
        -s|--subscription)
            SUBSCRIPTION_ID="$2"
            shift 2
            ;;
        -r|--release)
            HELM_RELEASE_NAME="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -l|--location)
            REGION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --skip-infra)
            SKIP_INFRA="true"
            shift
            ;;
        --force-recreate)
            FORCE_RECREATE="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validation
validate_environment() {
    case $ENVIRONMENT in
        development|staging|production)
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
}

validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check required tools
    local required_tools=("az" "kubectl" "helm" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check Azure CLI login
    if ! az account show &> /dev/null; then
        log_error "Azure CLI not logged in. Run 'az login' first"
        exit 1
    fi
    
    # Set subscription if provided
    if [[ -n "$SUBSCRIPTION_ID" ]]; then
        log_info "Setting Azure subscription to $SUBSCRIPTION_ID"
        az account set --subscription "$SUBSCRIPTION_ID"
    fi
    
    # Check Helm version
    local helm_version
    helm_version=$(helm version --short --client | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+')
    log_info "Using Helm version: $helm_version"
    
    # Check kubectl connectivity (if not skipping infra)
    if [[ "$SKIP_INFRA" == "false" ]] || kubectl cluster-info &> /dev/null; then
        log_info "Kubernetes cluster connectivity verified"
    fi
    
    log_success "Prerequisites validated"
}

# Infrastructure provisioning
provision_infrastructure() {
    if [[ "$SKIP_INFRA" == "true" ]]; then
        log_info "Skipping infrastructure provisioning"
        return 0
    fi
    
    log_info "Provisioning infrastructure for environment: $ENVIRONMENT"
    
    # Create resource group
    log_info "Creating resource group: $RESOURCE_GROUP"
    if [[ "$DRY_RUN" == "false" ]]; then
        az group create \
            --name "$RESOURCE_GROUP" \
            --location "$REGION" \
            --tags "environment=$ENVIRONMENT" "project=agentvault" \
            --output table
    fi
    
    # Deploy infrastructure using Terraform
    local terraform_dir="${PROJECT_ROOT}/infrastructure/terraform"
    if [[ -d "$terraform_dir" ]]; then
        log_info "Deploying Terraform infrastructure..."
        
        pushd "$terraform_dir" > /dev/null
        
        # Initialize Terraform
        terraform init \
            -backend-config="resource_group_name=${RESOURCE_GROUP}" \
            -backend-config="storage_account_name=agentvault${ENVIRONMENT}tf" \
            -backend-config="container_name=tfstate" \
            -backend-config="key=${ENVIRONMENT}.terraform.tfstate"
        
        # Plan infrastructure
        terraform plan \
            -var="environment=${ENVIRONMENT}" \
            -var="resource_group_name=${RESOURCE_GROUP}" \
            -var="location=${REGION}" \
            -var="cluster_name=${CLUSTER_NAME}" \
            -out="${ENVIRONMENT}.tfplan"
        
        # Apply infrastructure
        if [[ "$DRY_RUN" == "false" ]]; then
            terraform apply "${ENVIRONMENT}.tfplan"
        fi
        
        popd > /dev/null
    fi
    
    # Get AKS credentials
    log_info "Getting AKS credentials..."
    if [[ "$DRY_RUN" == "false" ]]; then
        az aks get-credentials \
            --resource-group "$RESOURCE_GROUP" \
            --name "$CLUSTER_NAME" \
            --overwrite-existing \
            --admin
    fi
    
    log_success "Infrastructure provisioned successfully"
}

# Kubernetes setup
setup_kubernetes() {
    log_info "Setting up Kubernetes resources..."
    
    # Create namespace
    log_info "Creating namespace: $NAMESPACE"
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    # Label namespace
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl label namespace "$NAMESPACE" \
            "environment=$ENVIRONMENT" \
            "managed-by=helm" \
            --overwrite
    fi
    
    # Install cert-manager (if not present)
    if ! kubectl get namespace cert-manager &> /dev/null; then
        log_info "Installing cert-manager..."
        if [[ "$DRY_RUN" == "false" ]]; then
            helm repo add jetstack https://charts.jetstack.io
            helm repo update
            
            helm upgrade --install cert-manager jetstack/cert-manager \
                --namespace cert-manager \
                --create-namespace \
                --version v1.13.0 \
                --set installCRDs=true \
                --wait
        fi
    fi
    
    # Install NGINX Ingress Controller (if not present)
    if ! kubectl get namespace ingress-nginx &> /dev/null; then
        log_info "Installing NGINX Ingress Controller..."
        if [[ "$DRY_RUN" == "false" ]]; then
            helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
            helm repo update
            
            helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
                --namespace ingress-nginx \
                --create-namespace \
                --set controller.replicaCount=2 \
                --set controller.nodeSelector."kubernetes\.io/os"=linux \
                --set defaultBackend.nodeSelector."kubernetes\.io/os"=linux \
                --wait
        fi
    fi
    
    # Install Azure NetApp Files CSI driver
    log_info "Installing Azure NetApp Files CSI driver..."
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/azurefile-csi-driver/master/deploy/example/storageclass-azurefile-csi.yaml
    fi
    
    # Install Secrets Store CSI Driver and Azure Key Vault Provider
    if ! kubectl get namespace kube-system | grep -q secrets-store-csi-driver; then
        log_info "Installing Secrets Store CSI Driver..."
        if [[ "$DRY_RUN" == "false" ]]; then
            helm repo add secrets-store-csi-driver https://kubernetes-sigs.github.io/secrets-store-csi-driver/charts
            helm repo update
            
            helm upgrade --install csi-secrets-store secrets-store-csi-driver/secrets-store-csi-driver \
                --namespace kube-system \
                --set syncSecret.enabled=true \
                --set enableSecretRotation=true \
                --wait
            
            # Install Azure Key Vault Provider
            kubectl apply -f https://raw.githubusercontent.com/Azure/secrets-store-csi-driver-provider-azure/master/deployment/provider-azure-installer.yaml
        fi
    fi
    
    log_success "Kubernetes setup completed"
}

# Helm deployment
deploy_helm_chart() {
    log_info "Deploying AgentVault using Helm..."
    
    # Add required Helm repositories
    log_info "Adding Helm repositories..."
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
    helm repo update
    
    # Prepare values file
    local values_file="${VALUES_DIR}/${ENVIRONMENT}.yaml"
    if [[ ! -f "$values_file" ]]; then
        log_warning "Values file not found: $values_file"
        log_info "Using default values from Chart"
        values_file=""
    else
        log_info "Using values file: $values_file"
    fi
    
    # Build Helm command
    local helm_cmd=(
        "helm" "upgrade" "--install" "$HELM_RELEASE_NAME" "$HELM_CHART_DIR"
        "--namespace" "$NAMESPACE"
        "--create-namespace"
        "--wait"
        "--timeout" "20m"
    )
    
    # Add values file if available
    if [[ -n "$values_file" ]]; then
        helm_cmd+=(--values "$values_file")
    fi
    
    # Add environment-specific overrides
    helm_cmd+=(
        --set "global.environment=$ENVIRONMENT"
        --set "image.tag=v1.0.0"
        --set "ingress.hosts[0].host=agentvault-${ENVIRONMENT}.yourdomain.com"
    )
    
    # Add dry-run flag if specified
    if [[ "$DRY_RUN" == "true" ]]; then
        helm_cmd+=(--dry-run --debug)
    fi
    
    # Execute Helm deployment
    log_info "Executing Helm deployment..."
    "${helm_cmd[@]}"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        log_success "AgentVault deployed successfully"
        
        # Wait for deployment to be ready
        log_info "Waiting for pods to be ready..."
        kubectl wait --for=condition=ready pod \
            --selector="app.kubernetes.io/name=agentvault" \
            --namespace="$NAMESPACE" \
            --timeout=300s
        
        # Display deployment status
        kubectl get all --namespace="$NAMESPACE"
        
        # Get ingress URL
        local ingress_ip
        ingress_ip=$(kubectl get ingress "$HELM_RELEASE_NAME" \
            --namespace="$NAMESPACE" \
            --output jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        
        if [[ -n "$ingress_ip" ]]; then
            log_success "AgentVault is accessible at: https://$ingress_ip"
        else
            log_info "Ingress IP not yet assigned. Check again in a few minutes."
        fi
    fi
}

# Post-deployment verification
verify_deployment() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Skipping verification for dry run"
        return 0
    fi
    
    log_info "Verifying deployment..."
    
    # Check pod status
    local failed_pods
    failed_pods=$(kubectl get pods --namespace="$NAMESPACE" \
        --field-selector=status.phase!=Running \
        --output jsonpath='{.items[*].metadata.name}' || echo "")
    
    if [[ -n "$failed_pods" ]]; then
        log_warning "Some pods are not running: $failed_pods"
        kubectl get pods --namespace="$NAMESPACE"
    else
        log_success "All pods are running"
    fi
    
    # Check services
    log_info "Checking services..."
    kubectl get services --namespace="$NAMESPACE"
    
    # Health check
    log_info "Performing health check..."
    local health_url="https://agentvault-${ENVIRONMENT}.yourdomain.com/health"
    if command -v curl &> /dev/null; then
        if curl -f -s "$health_url" > /dev/null; then
            log_success "Health check passed"
        else
            log_warning "Health check failed - service may still be starting"
        fi
    fi
    
    log_success "Deployment verification completed"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            log_info "Collecting debug information..."
            kubectl get events --namespace="$NAMESPACE" --sort-by='.lastTimestamp' | tail -20
            kubectl logs --namespace="$NAMESPACE" --selector="app.kubernetes.io/name=agentvault" --tail=50
        fi
    fi
}

# Main execution
main() {
    log_info "Starting AgentVault deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Cluster: $CLUSTER_NAME"
    log_info "Resource Group: $RESOURCE_GROUP"
    log_info "Namespace: $NAMESPACE"
    log_info "Region: $REGION"
    log_info "Dry Run: $DRY_RUN"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Execute deployment steps
    validate_environment
    validate_prerequisites
    
    if [[ "$FORCE_RECREATE" == "true" ]]; then
        log_warning "Force recreate enabled - this will delete existing resources"
        if [[ "$DRY_RUN" == "false" ]]; then
            helm uninstall "$HELM_RELEASE_NAME" --namespace="$NAMESPACE" || true
            kubectl delete namespace "$NAMESPACE" || true
        fi
    fi
    
    provision_infrastructure
    setup_kubernetes
    deploy_helm_chart
    verify_deployment
    
    log_success "AgentVault deployment completed successfully!"
}

# Execute main function
main "$@"