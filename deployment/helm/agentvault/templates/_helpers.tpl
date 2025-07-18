{{/*
AgentVault™ Helm Chart Template Helpers
Common template functions and naming conventions
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "agentvault.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "agentvault.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "agentvault.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "agentvault.labels" -}}
helm.sh/chart: {{ include "agentvault.chart" . }}
{{ include "agentvault.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: core
app.kubernetes.io/part-of: agentvault
{{- end }}

{{/*
Selector labels
*/}}
{{- define "agentvault.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agentvault.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "agentvault.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "agentvault.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the orchestrator service account
*/}}
{{- define "agentvault.orchestrator.serviceAccountName" -}}
{{- if .Values.orchestrator.serviceAccount.create }}
{{- default (printf "%s-orchestrator" (include "agentvault.fullname" .)) .Values.orchestrator.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.orchestrator.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create orchestrator labels
*/}}
{{- define "agentvault.orchestrator.labels" -}}
helm.sh/chart: {{ include "agentvault.chart" . }}
{{ include "agentvault.orchestrator.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: orchestrator
app.kubernetes.io/part-of: agentvault
{{- end }}

{{/*
Orchestrator selector labels
*/}}
{{- define "agentvault.orchestrator.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agentvault.name" . }}-orchestrator
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create ML services labels
*/}}
{{- define "agentvault.mlservices.labels" -}}
helm.sh/chart: {{ include "agentvault.chart" . }}
{{ include "agentvault.mlservices.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: ml-services
app.kubernetes.io/part-of: agentvault
{{- end }}

{{/*
ML services selector labels
*/}}
{{- define "agentvault.mlservices.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agentvault.name" . }}-ml-services
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create vector DB labels
*/}}
{{- define "agentvault.vectordb.labels" -}}
helm.sh/chart: {{ include "agentvault.chart" . }}
{{ include "agentvault.vectordb.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: vector-database
app.kubernetes.io/part-of: agentvault
{{- end }}

{{/*
Vector DB selector labels
*/}}
{{- define "agentvault.vectordb.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agentvault.name" . }}-vector-db
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Generate certificates
*/}}
{{- define "agentvault.gen-certs" -}}
{{- $altNames := list ( printf "%s.%s" (include "agentvault.fullname" .) .Release.Namespace ) ( printf "%s.%s.svc" (include "agentvault.fullname" .) .Release.Namespace ) -}}
{{- $ca := genCA "agentvault-ca" 3650 -}}
{{- $cert := genSignedCert ( include "agentvault.fullname" . ) nil $altNames 3650 $ca -}}
tls.crt: {{ $cert.Cert | b64enc }}
tls.key: {{ $cert.Key | b64enc }}
ca.crt: {{ $ca.Cert | b64enc }}
{{- end }}

{{/*
Create Azure NetApp Files capacity pool name
*/}}
{{- define "agentvault.anf.capacityPool" -}}
{{- $tier := . -}}
{{- printf "agentvault-%s-pool" $tier }}
{{- end }}

{{/*
Create Azure NetApp Files volume name
*/}}
{{- define "agentvault.anf.volume" -}}
{{- $context := index . 0 -}}
{{- $tier := index . 1 -}}
{{- printf "%s-%s-volume" (include "agentvault.fullname" $context) $tier }}
{{- end }}

{{/*
Create persistent volume claim name for ANF
*/}}
{{- define "agentvault.anf.pvc" -}}
{{- $context := index . 0 -}}
{{- $tier := index . 1 -}}
{{- printf "%s-anf-%s" (include "agentvault.fullname" $context) $tier }}
{{- end }}

{{/*
Generate Redis connection string
*/}}
{{- define "agentvault.redis.connectionString" -}}
{{- if .Values.redis.enabled -}}
{{- if .Values.redis.auth.enabled -}}
redis://{{ .Values.redis.auth.username | default "default" }}:{{ .Values.redis.auth.password }}@{{ include "agentvault.fullname" . }}-redis-master:6379/0
{{- else -}}
redis://{{ include "agentvault.fullname" . }}-redis-master:6379/0
{{- end -}}
{{- else -}}
{{ .Values.externalRedis.connectionString }}
{{- end -}}
{{- end }}

{{/*
Generate PostgreSQL connection string
*/}}
{{- define "agentvault.postgresql.connectionString" -}}
{{- if .Values.postgresql.enabled -}}
postgresql://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ include "agentvault.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- else -}}
{{ .Values.externalPostgreSQL.connectionString }}
{{- end -}}
{{- end }}

{{/*
Create secret name for TLS certificates
*/}}
{{- define "agentvault.secretName" -}}
{{- printf "%s-tls" (include "agentvault.fullname" .) }}
{{- end }}

{{/*
Create config map name
*/}}
{{- define "agentvault.configMapName" -}}
{{- printf "%s-config" (include "agentvault.fullname" .) }}
{{- end }}

{{/*
Create secret name for application secrets
*/}}
{{- define "agentvault.secretAppName" -}}
{{- printf "%s-secrets" (include "agentvault.fullname" .) }}
{{- end }}

{{/*
Validate required values
*/}}
{{- define "agentvault.validateValues" -}}
{{- if and .Values.anf.enabled (not .Values.anf.subscriptionId) -}}
{{- fail "ANF is enabled but subscriptionId is not set" -}}
{{- end -}}
{{- if and .Values.anf.enabled (not .Values.anf.resourceGroup) -}}
{{- fail "ANF is enabled but resourceGroup is not set" -}}
{{- end -}}
{{- if and .Values.security.encryption.enabled (not .Values.security.encryption.keyVaultName) -}}
{{- fail "Encryption is enabled but keyVaultName is not set" -}}
{{- end -}}
{{- end }}

{{/*
Create environment variables for ANF configuration
*/}}
{{- define "agentvault.anf.env" -}}
- name: ANF_SUBSCRIPTION_ID
  value: {{ .Values.anf.subscriptionId | quote }}
- name: ANF_RESOURCE_GROUP
  value: {{ .Values.anf.resourceGroup | quote }}
- name: ANF_ACCOUNT_NAME
  value: {{ .Values.anf.accountName | quote }}
- name: ANF_LOCATION
  value: {{ .Values.anf.location | quote }}
{{- range $tier, $config := .Values.anf.capacityPools }}
- name: ANF_{{ upper $tier }}_POOL
  value: {{ $config.name | quote }}
{{- end }}
{{- end }}

{{/*
Create environment variables for security configuration
*/}}
{{- define "agentvault.security.env" -}}
{{- if .Values.security.encryption.enabled }}
- name: ENCRYPTION_ENABLED
  value: "true"
- name: ENCRYPTION_PROVIDER
  value: {{ .Values.security.encryption.provider | quote }}
- name: KEY_VAULT_NAME
  value: {{ .Values.security.encryption.keyVaultName | quote }}
- name: KEY_ROTATION_INTERVAL
  value: {{ .Values.security.encryption.keyRotationInterval | quote }}
{{- end }}
{{- if .Values.security.authentication.enabled }}
- name: AUTHENTICATION_ENABLED
  value: "true"
{{- if has "azure-ad" .Values.security.authentication.providers }}
- name: AZURE_AD_TENANT_ID
  valueFrom:
    secretKeyRef:
      name: {{ include "agentvault.secretAppName" . }}
      key: azure-ad-tenant-id
- name: AZURE_AD_CLIENT_ID
  valueFrom:
    secretKeyRef:
      name: {{ include "agentvault.secretAppName" . }}
      key: azure-ad-client-id
- name: AZURE_AD_CLIENT_SECRET
  valueFrom:
    secretKeyRef:
      name: {{ include "agentvault.secretAppName" . }}
      key: azure-ad-client-secret
{{- end }}
{{- end }}
{{- end }}

{{/*
Create environment variables for monitoring configuration
*/}}
{{- define "agentvault.monitoring.env" -}}
{{- if .Values.monitoring.enabled }}
- name: MONITORING_ENABLED
  value: "true"
- name: METRICS_PORT
  value: "8081"
- name: METRICS_PATH
  value: "/metrics"
{{- if .Values.monitoring.customMetrics.azureMonitor.enabled }}
- name: AZURE_MONITOR_ENABLED
  value: "true"
- name: AZURE_MONITOR_WORKSPACE_ID
  value: {{ .Values.monitoring.customMetrics.azureMonitor.workspaceId | quote }}
- name: DATA_COLLECTION_ENDPOINT
  value: {{ .Values.monitoring.customMetrics.azureMonitor.dataCollectionEndpoint | quote }}
- name: DATA_COLLECTION_RULE_ID
  value: {{ .Values.monitoring.customMetrics.azureMonitor.dataCollectionRuleId | quote }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create resource limits for GPU workloads
*/}}
{{- define "agentvault.gpu.resources" -}}
{{- if .gpuEnabled }}
limits:
  nvidia.com/gpu: {{ .gpuCount | default 1 }}
  cpu: {{ .cpuLimit | default "2000m" }}
  memory: {{ .memoryLimit | default "4Gi" }}
requests:
  cpu: {{ .cpuRequest | default "1000m" }}
  memory: {{ .memoryRequest | default "2Gi" }}
{{- else }}
limits:
  cpu: {{ .cpuLimit | default "1000m" }}
  memory: {{ .memoryLimit | default "2Gi" }}
requests:
  cpu: {{ .cpuRequest | default "500m" }}
  memory: {{ .memoryRequest | default "1Gi" }}
{{- end }}
{{- end }}