"""
AgentVault™ + AutoGen Multi-Agent Healthcare System
HIPAA-Compliant Medical Diagnosis and Treatment Planning

This example demonstrates a sophisticated multi-agent healthcare system using
AutoGen and AgentVault™ for HIPAA-compliant enterprise storage.

Agents:
- Primary Care Physician Agent
- Radiologist Agent  
- Specialist Consultant Agent
- Medical Records Agent
- Treatment Planning Agent

Features:
- HIPAA-compliant patient data storage
- Real-time medical image analysis with vector search
- Multi-agent medical consultation workflows
- Complete audit trail for regulatory compliance
- 99.99% uptime SLA for critical healthcare operations

Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json
import hashlib
from dataclasses import dataclass

import autogen
from autogen import GroupChat, GroupChatManager, UserProxyAgent, AssistantAgent
from autogen.agentchat.contrib.capabilities.teachability import Teachability

from agentvault import AgentVaultOrchestrator
from agentvault.core.storage_orchestrator import StorageRequest, StorageTier


@dataclass
class PatientData:
    """HIPAA-compliant patient data structure"""
    patient_id: str
    age: int
    gender: str
    medical_history: List[str]
    current_symptoms: List[str]
    vital_signs: Dict[str, Any]
    medications: List[str]
    allergies: List[str]
    insurance_info: Dict[str, str]
    emergency_contact: Dict[str, str]
    
    def get_anonymized_id(self) -> str:
        """Get anonymized patient ID for non-PHI operations"""
        return hashlib.sha256(self.patient_id.encode()).hexdigest()[:16]


class AgentVaultAutogenAgent:
    """Base class for AutoGen agents with AgentVault™ integration"""
    
    def __init__(self, name: str, agent_type: str, orchestrator: AgentVaultOrchestrator):
        self.name = name
        self.agent_type = agent_type
        self.orchestrator = orchestrator
        self.agent_id = f"healthcare-{name.lower().replace(' ', '-')}"
        
        # Performance tracking
        self.interactions = 0
        self.total_latency = 0.0
    
    async def store_interaction(self, message: str, response: str, 
                              patient_id: str = None, phi_level: str = "low") -> None:
        """Store agent interaction with appropriate security level"""
        
        # Determine storage tier based on PHI level
        tier = StorageTier.PREMIUM if phi_level == "high" else StorageTier.STANDARD
        
        storage_request = StorageRequest(
            agent_id=self.agent_id,
            operation="write",
            data_type="activity_log",
            priority="high" if phi_level == "high" else "normal",
            latency_requirement=1.0,
            encryption_required=True,
            compliance_tags=["HIPAA", "PHI", "healthcare", "audit_trail"],
            metadata={
                "interaction_type": "agent_conversation",
                "message": message,
                "response": response,
                "patient_id_hash": hashlib.sha256(patient_id.encode()).hexdigest() if patient_id else None,
                "phi_level": phi_level,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_name": self.name
            }
        )
        
        await self.orchestrator.process_storage_request(storage_request)
    
    async def retrieve_patient_context(self, patient_id: str) -> Dict[str, Any]:
        """Retrieve patient context with HIPAA compliance"""
        
        storage_request = StorageRequest(
            agent_id=self.agent_id,
            operation="read",
            data_type="long_term_memory",
            priority="high",
            latency_requirement=0.5,
            encryption_required=True,
            compliance_tags=["HIPAA", "PHI", "patient_data"],
            metadata={
                "query_type": "patient_context",
                "patient_id_hash": hashlib.sha256(patient_id.encode()).hexdigest(),
                "requested_by": self.agent_id
            }
        )
        
        result = await self.orchestrator.process_storage_request(storage_request)
        return result.get('result', {})


class PrimaryCarePhysicianAgent(AgentVaultAutogenAgent):
    """Primary Care Physician Agent with AgentVault™ integration"""
    
    def __init__(self, orchestrator: AgentVaultOrchestrator):
        super().__init__("Primary Care Physician", "autogen", orchestrator)
        
        self.autogen_agent = AssistantAgent(
            name="PrimaryCarePhysician",
            system_message="""You are a Primary Care Physician with 15 years of experience. 
            You conduct initial patient assessments, coordinate care with specialists, and make treatment decisions.
            
            Guidelines:
            - Always maintain HIPAA compliance
            - Ask relevant medical history questions  
            - Consider differential diagnoses
            - Coordinate with specialists when needed
            - Provide clear, evidence-based recommendations
            - Document all interactions for medical records
            
            You have access to:
            - Complete patient medical history via AgentVault™
            - Real-time consultation with specialists
            - Medical knowledge base with latest research
            - Treatment guidelines and protocols
            """,
            llm_config={
                "model": "gpt-4",
                "temperature": 0.3,  # Conservative for medical decisions
                "timeout": 60
            }
        )
    
    async def assess_patient(self, patient_data: PatientData) -> Dict[str, Any]:
        """Perform initial patient assessment"""
        
        start_time = datetime.utcnow()
        
        try:
            # Retrieve patient history from AgentVault™ premium storage
            patient_context = await self.retrieve_patient_context(patient_data.patient_id)
            
            # Combine current symptoms with historical data
            assessment_prompt = f"""
            Patient Assessment Request:
            
            Current Symptoms: {', '.join(patient_data.current_symptoms)}
            Vital Signs: {json.dumps(patient_data.vital_signs, indent=2)}
            Medical History: {', '.join(patient_data.medical_history)}
            Current Medications: {', '.join(patient_data.medications)}
            Allergies: {', '.join(patient_data.allergies)}
            
            Historical Context: {json.dumps(patient_context, indent=2)}
            
            Please provide:
            1. Initial assessment and differential diagnosis
            2. Recommended diagnostic tests
            3. Specialist referrals if needed
            4. Immediate treatment recommendations
            5. Risk assessment and monitoring plan
            """
            
            # Generate assessment
            response = self.autogen_agent.generate_reply(
                messages=[{"role": "user", "content": assessment_prompt}]
            )
            
            # Calculate latency
            latency = (datetime.utcnow() - start_time).total_seconds()
            
            # Store interaction for medical records and compliance
            await self.store_interaction(
                message=assessment_prompt,
                response=response,
                patient_id=patient_data.patient_id,
                phi_level="high"
            )
            
            return {
                "assessment": response,
                "physician": self.name,
                "timestamp": start_time.isoformat(),
                "latency_ms": latency * 1000,
                "patient_id": patient_data.get_anonymized_id(),
                "compliance_logged": True
            }
            
        except Exception as e:
            logging.error(f"Patient assessment failed: {e}")
            return {"error": str(e), "physician": self.name}


class RadiologistAgent(AgentVaultAutogenAgent):
    """Radiologist Agent for medical imaging analysis"""
    
    def __init__(self, orchestrator: AgentVaultOrchestrator):
        super().__init__("Radiologist", "autogen", orchestrator)
        
        self.autogen_agent = AssistantAgent(
            name="Radiologist",
            system_message="""You are a Board-Certified Radiologist specializing in diagnostic imaging.
            You analyze medical images (X-rays, CT, MRI, ultrasound) and provide detailed radiological reports.
            
            Expertise:
            - Diagnostic imaging interpretation
            - Cross-sectional anatomy
            - Pathology recognition
            - Intervention guidance
            - Emergency radiology
            
            You have access to:
            - Ultra-fast medical image database via AgentVault™
            - AI-powered image analysis tools  
            - Historical imaging studies for comparison
            - Medical imaging knowledge base
            - PACS integration for image retrieval
            """,
            llm_config={
                "model": "gpt-4",
                "temperature": 0.2,  # Very conservative for medical imaging
                "timeout": 120
            }
        )
    
    async def analyze_medical_images(self, patient_id: str, 
                                   imaging_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze medical images with ultra-fast vector search"""
        
        start_time = datetime.utcnow()
        
        try:
            # Use AgentVault™ ultra-performance tier for image vector search
            storage_request = StorageRequest(
                agent_id=self.agent_id,
                operation="query",
                data_type="vector",
                priority="critical",
                latency_requirement=0.1,  # 100ms for emergency cases
                encryption_required=True,
                compliance_tags=["HIPAA", "medical_imaging", "diagnostic"],
                metadata={
                    "query_type": "image_similarity_search",
                    "imaging_modality": imaging_data.get("modality", "unknown"),
                    "body_region": imaging_data.get("region", "unknown"),
                    "clinical_indication": imaging_data.get("indication", "unknown")
                }
            )
            
            # Execute ultra-fast image vector search
            search_result = await self.orchestrator.process_storage_request(storage_request)
            
            # Generate radiological interpretation
            analysis_prompt = f"""
            Radiological Analysis Request:
            
            Imaging Study: {imaging_data.get('modality', 'Unknown')}
            Body Region: {imaging_data.get('region', 'Unknown')}
            Clinical Indication: {imaging_data.get('indication', 'Unknown')}
            
            Image Characteristics:
            {json.dumps(imaging_data.get('characteristics', {}), indent=2)}
            
            Similar Cases Found: {len(search_result.get('result', {}).get('similar_cases', []))}
            
            Please provide:
            1. Detailed radiological findings
            2. Differential diagnosis based on imaging
            3. Comparison with similar historical cases
            4. Recommendations for additional imaging if needed
            5. Urgency level and follow-up requirements
            6. Structured report in standard radiological format
            """
            
            response = self.autogen_agent.generate_reply(
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            latency = (datetime.utcnow() - start_time).total_seconds()
            
            # Store radiological report
            await self.store_interaction(
                message=analysis_prompt,
                response=response,
                patient_id=patient_id,
                phi_level="high"
            )
            
            return {
                "radiological_report": response,
                "radiologist": self.name,
                "imaging_modality": imaging_data.get('modality'),
                "timestamp": start_time.isoformat(),
                "latency_ms": latency * 1000,
                "similar_cases_found": len(search_result.get('result', {}).get('similar_cases', [])),
                "vector_search_latency": search_result.get('duration_ms', 0),
                "compliance_logged": True
            }
            
        except Exception as e:
            logging.error(f"Medical image analysis failed: {e}")
            return {"error": str(e), "radiologist": self.name}


class SpecialistConsultantAgent(AgentVaultAutogenAgent):
    """Specialist Consultant Agent for expert medical opinions"""
    
    def __init__(self, specialty: str, orchestrator: AgentVaultOrchestrator):
        super().__init__(f"{specialty} Specialist", "autogen", orchestrator)
        self.specialty = specialty
        
        self.autogen_agent = AssistantAgent(
            name=f"{specialty}Specialist",
            system_message=f"""You are a Board-Certified {specialty} with extensive clinical experience.
            You provide expert consultation and specialized treatment recommendations.
            
            Your expertise includes:
            - Advanced {specialty.lower()} diagnostics
            - Complex case management  
            - Latest treatment protocols
            - Evidence-based medicine
            - Multidisciplinary care coordination
            
            Guidelines:
            - Provide specialized expertise beyond general medicine
            - Consider latest research and guidelines
            - Assess risks and benefits of treatments
            - Coordinate with primary care and other specialists
            - Document detailed consultation notes
            """,
            llm_config={
                "model": "gpt-4",
                "temperature": 0.25,
                "timeout": 90
            }
        )
    
    async def provide_consultation(self, case_summary: str, 
                                 specific_question: str, patient_id: str) -> Dict[str, Any]:
        """Provide specialist consultation"""
        
        start_time = datetime.utcnow()
        
        try:
            # Retrieve specialist knowledge base
            knowledge_request = StorageRequest(
                agent_id=self.agent_id,
                operation="query", 
                data_type="knowledge_graph",
                priority="high",
                latency_requirement=2.0,
                metadata={
                    "specialty": self.specialty,
                    "query_type": "specialist_consultation",
                    "case_complexity": "standard"
                }
            )
            
            knowledge_result = await self.orchestrator.process_storage_request(knowledge_request)
            
            consultation_prompt = f"""
            Specialist Consultation Request - {self.specialty}
            
            Case Summary:
            {case_summary}
            
            Specific Question:
            {specific_question}
            
            Relevant Knowledge Base:
            {json.dumps(knowledge_result.get('result', {}), indent=2)}
            
            Please provide:
            1. Specialist assessment and opinion
            2. Recommended diagnostic workup specific to {self.specialty.lower()}
            3. Treatment options and recommendations
            4. Prognosis and expected outcomes
            5. Follow-up care plan
            6. Coordination recommendations for primary care team
            """
            
            response = self.autogen_agent.generate_reply(
                messages=[{"role": "user", "content": consultation_prompt}]
            )
            
            latency = (datetime.utcnow() - start_time).total_seconds()
            
            # Store consultation
            await self.store_interaction(
                message=consultation_prompt,
                response=response,
                patient_id=patient_id,
                phi_level="high"
            )
            
            return {
                "consultation": response,
                "specialist": self.name,
                "specialty": self.specialty,
                "timestamp": start_time.isoformat(),
                "latency_ms": latency * 1000,
                "knowledge_base_accessed": True,
                "compliance_logged": True
            }
            
        except Exception as e:
            logging.error(f"Specialist consultation failed: {e}")
            return {"error": str(e), "specialist": self.name}


class HealthcareMultiAgentSystem:
    """
    HIPAA-Compliant Healthcare Multi-Agent System using AgentVault™
    
    This system orchestrates multiple AI agents for comprehensive healthcare:
    - Primary Care Physician for initial assessment
    - Radiologist for medical imaging analysis
    - Multiple Specialists for expert consultation
    - Treatment Planning coordination
    - Complete compliance and audit trail
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # AgentVault™ orchestrator
        self.orchestrator = None
        
        # Medical agents
        self.primary_care_agent = None
        self.radiologist_agent = None
        self.specialists = {}
        
        # AutoGen group chat
        self.group_chat = None
        self.group_chat_manager = None
        
        # Case tracking
        self.active_cases = {}
        self.consultation_history = []
    
    async def initialize(self) -> None:
        """Initialize the healthcare multi-agent system"""
        
        try:
            self.logger.info("Initializing Healthcare Multi-Agent System with AgentVault™...")
            
            # Initialize AgentVault™ with healthcare-specific configuration
            self.orchestrator = AgentVaultOrchestrator(self.config)
            await self.orchestrator.initialize()
            
            # Initialize medical agents
            self.primary_care_agent = PrimaryCarePhysicianAgent(self.orchestrator)
            self.radiologist_agent = RadiologistAgent(self.orchestrator)
            
            # Initialize specialist agents
            specialties = ["Cardiology", "Neurology", "Oncology", "Endocrinology"]
            for specialty in specialties:
                self.specialists[specialty] = SpecialistConsultantAgent(specialty, self.orchestrator)
            
            # Setup AutoGen group chat
            agents = [
                self.primary_care_agent.autogen_agent,
                self.radiologist_agent.autogen_agent
            ] + [specialist.autogen_agent for specialist in self.specialists.values()]
            
            self.group_chat = GroupChat(
                agents=agents,
                messages=[],
                max_round=20,
                speaker_selection_method="round_robin"
            )
            
            self.group_chat_manager = GroupChatManager(
                groupchat=self.group_chat,
                llm_config={
                    "model": "gpt-4",
                    "temperature": 0.1
                }
            )
            
            self.logger.info("Healthcare Multi-Agent System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize healthcare system: {e}")
            raise
    
    async def process_patient_case(self, patient_data: PatientData, 
                                 imaging_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process complete patient case with multi-agent consultation"""
        
        case_id = f"case-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{patient_data.get_anonymized_id()[:8]}"
        self.logger.info(f"Processing patient case: {case_id}")
        
        try:
            case_results = {
                "case_id": case_id,
                "patient_id": patient_data.get_anonymized_id(),
                "start_time": datetime.utcnow().isoformat(),
                "agents_consulted": [],
                "consultations": {},
                "timeline": []
            }
            
            # Step 1: Primary Care Assessment
            self.logger.info("Step 1: Primary care assessment")
            primary_assessment = await self.primary_care_agent.assess_patient(patient_data)
            case_results["consultations"]["primary_care"] = primary_assessment
            case_results["agents_consulted"].append("Primary Care Physician")
            case_results["timeline"].append({
                "step": "primary_assessment",
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "Primary Care Physician",
                "duration_ms": primary_assessment.get("latency_ms", 0)
            })
            
            # Step 2: Radiological Analysis (if imaging data provided)
            if imaging_data:
                self.logger.info("Step 2: Radiological analysis")
                radiology_report = await self.radiologist_agent.analyze_medical_images(
                    patient_data.patient_id, imaging_data
                )
                case_results["consultations"]["radiology"] = radiology_report
                case_results["agents_consulted"].append("Radiologist")
                case_results["timeline"].append({
                    "step": "radiology_analysis", 
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": "Radiologist",
                    "duration_ms": radiology_report.get("latency_ms", 0)
                })
            
            # Step 3: Determine needed specialist consultations
            needed_specialties = self._determine_needed_specialties(
                primary_assessment, case_results.get("consultations", {}).get("radiology")
            )
            
            # Step 4: Specialist Consultations
            for specialty in needed_specialties:
                if specialty in self.specialists:
                    self.logger.info(f"Step 4: {specialty} consultation")
                    
                    case_summary = self._generate_case_summary(case_results, patient_data)
                    specialist_question = f"Please provide {specialty.lower()} consultation for this case."
                    
                    consultation = await self.specialists[specialty].provide_consultation(
                        case_summary, specialist_question, patient_data.patient_id
                    )
                    
                    case_results["consultations"][specialty.lower()] = consultation
                    case_results["agents_consulted"].append(f"{specialty} Specialist")
                    case_results["timeline"].append({
                        "step": f"{specialty.lower()}_consultation",
                        "timestamp": datetime.utcnow().isoformat(),
                        "agent": f"{specialty} Specialist",
                        "duration_ms": consultation.get("latency_ms", 0)
                    })
            
            # Step 5: Generate comprehensive treatment plan
            treatment_plan = await self._generate_treatment_plan(case_results, patient_data)
            case_results["treatment_plan"] = treatment_plan
            
            # Step 6: Store complete case for medical records
            await self._store_complete_case(case_results, patient_data)
            
            case_results["end_time"] = datetime.utcnow().isoformat()
            case_results["total_agents"] = len(case_results["agents_consulted"])
            case_results["status"] = "completed"
            
            self.logger.info(f"Patient case {case_id} processed successfully")
            return case_results
            
        except Exception as e:
            self.logger.error(f"Failed to process patient case: {e}")
            return {
                "case_id": case_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _determine_needed_specialties(self, primary_assessment: Dict[str, Any], 
                                    radiology_report: Optional[Dict[str, Any]] = None) -> List[str]:
        """Determine which specialist consultations are needed"""
        
        # Simple rule-based logic for demo
        # In production, this would use NLP to analyze assessment text
        
        needed = []
        assessment_text = primary_assessment.get("assessment", "").lower()
        
        if any(keyword in assessment_text for keyword in ["heart", "cardiac", "chest pain", "ecg"]):
            needed.append("Cardiology")
        
        if any(keyword in assessment_text for keyword in ["neurologic", "headache", "seizure", "stroke"]):
            needed.append("Neurology")
        
        if any(keyword in assessment_text for keyword in ["diabetes", "thyroid", "hormone", "endocrine"]):
            needed.append("Endocrinology")
        
        if any(keyword in assessment_text for keyword in ["cancer", "tumor", "malignancy", "oncology"]):
            needed.append("Oncology")
        
        return needed[:2]  # Limit to 2 specialists for demo
    
    def _generate_case_summary(self, case_results: Dict[str, Any], 
                             patient_data: PatientData) -> str:
        """Generate comprehensive case summary for specialists"""
        
        summary = f"""
        Patient Case Summary:
        
        Demographics: {patient_data.age} year old {patient_data.gender}
        
        Chief Complaint: {', '.join(patient_data.current_symptoms)}
        
        Medical History: {', '.join(patient_data.medical_history)}
        
        Current Medications: {', '.join(patient_data.medications)}
        
        Allergies: {', '.join(patient_data.allergies)}
        
        Vital Signs: {json.dumps(patient_data.vital_signs, indent=2)}
        
        Primary Care Assessment:
        {case_results.get('consultations', {}).get('primary_care', {}).get('assessment', 'Not available')}
        """
        
        if 'radiology' in case_results.get('consultations', {}):
            summary += f"""
        
        Radiology Report:
        {case_results['consultations']['radiology'].get('radiological_report', 'Not available')}
        """
        
        return summary
    
    async def _generate_treatment_plan(self, case_results: Dict[str, Any], 
                                     patient_data: PatientData) -> Dict[str, Any]:
        """Generate comprehensive treatment plan based on all consultations"""
        
        # Aggregate all consultation findings
        all_consultations = case_results.get("consultations", {})
        
        treatment_plan = {
            "patient_id": patient_data.get_anonymized_id(),
            "generated_at": datetime.utcnow().isoformat(),
            "primary_diagnosis": "To be determined based on consultations",
            "differential_diagnoses": [],
            "treatment_recommendations": [],
            "follow_up_plan": [],
            "specialist_referrals": [],
            "monitoring_requirements": [],
            "patient_education": [],
            "estimated_timeline": "To be determined"
        }
        
        # This would be more sophisticated in production
        treatment_plan["summary"] = f"Comprehensive treatment plan based on {len(all_consultations)} specialist consultations"
        
        return treatment_plan
    
    async def _store_complete_case(self, case_results: Dict[str, Any], 
                                 patient_data: PatientData) -> None:
        """Store complete case in HIPAA-compliant storage"""
        
        storage_request = StorageRequest(
            agent_id="healthcare-case-manager",
            operation="write",
            data_type="long_term_memory",
            priority="high",
            latency_requirement=5.0,
            encryption_required=True,
            compliance_tags=["HIPAA", "PHI", "medical_records", "case_history"],
            metadata={
                "case_type": "multi_agent_consultation",
                "patient_id_hash": patient_data.get_anonymized_id(),
                "case_data": case_results,
                "storage_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        await self.orchestrator.process_storage_request(storage_request)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup HIPAA-compliant logging"""
        
        logger = logging.getLogger("agentvault.healthcare_system")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger


# Example usage
async def main():
    """Example usage of the Healthcare Multi-Agent System"""
    
    # Configuration for healthcare deployment
    config = {
        "azure": {
            "subscription_id": "your-subscription-id",
            "resource_group": "agentvault-healthcare-rg",
            "location": "East US 2"
        },
        "anf": {
            "account_name": "agentvault-healthcare-anf",
            "subnet_id": "/subscriptions/.../subnets/healthcare-subnet"
        },
        "redis": {
            "host": "agentvault-healthcare-redis.redis.cache.windows.net",
            "port": 6380,
            "password": "your-redis-password"
        },
        "security": {
            "encryption_enabled": True,
            "compliance_level": "HIPAA",
            "audit_enabled": True
        }
    }
    
    # Initialize healthcare system
    healthcare_system = HealthcareMultiAgentSystem(config)
    await healthcare_system.initialize()
    
    # Example patient case
    patient = PatientData(
        patient_id="PATIENT-12345",
        age=45,
        gender="Female",
        medical_history=["Hypertension", "Type 2 Diabetes", "Hyperlipidemia"],
        current_symptoms=["Chest pain", "Shortness of breath", "Fatigue"],
        vital_signs={
            "blood_pressure": "150/95",
            "heart_rate": 95,
            "temperature": 98.6,
            "oxygen_saturation": 96,
            "respiratory_rate": 18
        },
        medications=["Lisinopril 10mg", "Metformin 1000mg", "Atorvastatin 20mg"],
        allergies=["Penicillin", "Shellfish"],
        insurance_info={"provider": "Blue Cross", "id": "BC123456"},
        emergency_contact={"name": "John Doe", "phone": "555-0123"}
    )
    
    # Example imaging data
    imaging_data = {
        "modality": "Chest X-Ray",
        "region": "Thorax",
        "indication": "Chest pain and shortness of breath",
        "characteristics": {
            "heart_size": "normal",
            "lung_fields": "clear", 
            "mediastinum": "normal",
            "bones": "intact"
        }
    }
    
    # Process patient case
    print("Processing patient case with multi-agent consultation...")
    case_result = await healthcare_system.process_patient_case(patient, imaging_data)
    
    print("\\nCase Processing Complete!")
    print(f"Case ID: {case_result['case_id']}")
    print(f"Agents Consulted: {', '.join(case_result['agents_consulted'])}")
    print(f"Total Consultations: {len(case_result.get('consultations', {}))}")
    print(f"Processing Status: {case_result['status']}")
    
    # Display consultation results
    for consultation_type, result in case_result.get('consultations', {}).items():
        print(f"\\n{consultation_type.title()} Consultation:")
        print(f"  Latency: {result.get('latency_ms', 0):.1f}ms")
        print(f"  Compliance Logged: {result.get('compliance_logged', False)}")


if __name__ == "__main__":
    asyncio.run(main())