"""
NEXUS CDM Configuration
=======================
Neural EXpert Unified System for Clinical Data Management

Configuration settings for API keys, data paths, and system parameters.
"""

import os
from dataclasses import dataclass
from typing import Dict, List

# OpenAI API Configuration - Use Streamlit Secrets
import streamlit as st
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_MODEL = "gpt-5.2"  # Using GPT-5.2 with Responses API

# Data Paths - relative to repo root for cloud deployment
DATA_BASE_PATH = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "outputs")

@dataclass
class DQIWeightConfig:
    """
    Data Quality Index (DQI) Weight Configuration

    Weights are derived from ICH E6(R3) regulatory requirements and patient safety priorities.
    Total weights = 100%

    References:
    - SAE: ICH E2A, E6(R3) - Direct patient safety impact
    - Visits: FDA 21 CFR 312.62 - Safety monitoring compliance
    - Queries: ICH E6(R3) §5.18.4 - Data integrity (ALCOA)
    - CRF: ICH E6(R3) §8 - Essential document compliance
    - Lab: ICH E6(R3) §5.18.3 - Safety data completeness
    - PD: ICH E6(R3) §4.5 - GCP compliance
    - Coding: FDA Data Standards Catalog - Submission readiness
    - Signatures: 21 CFR Part 11 - PI oversight verification
    - SDV: ICH E6(R3) §5.18.4 - Source verification
    """
    sae_reconciliation: float = 0.18      # CRITICAL - Direct patient safety
    missing_visits: float = 0.15          # HIGH - Safety monitoring gaps
    open_queries: float = 0.15            # HIGH - Data integrity (ALCOA)
    crf_completion: float = 0.12          # MEDIUM - Essential documents
    lab_integrity: float = 0.12           # MEDIUM - Safety data
    protocol_deviations: float = 0.10     # MEDIUM - GCP compliance
    coding_completion: float = 0.08       # LOW - Regulatory submission
    signature_compliance: float = 0.05    # LOW - PI oversight
    sdv_completion: float = 0.05          # LOW - Source verification


@dataclass
class QTLConfig:
    """
    Quality Tolerance Limits (QTL) Configuration

    Per ICH E6(R3) RBQM requirements, these thresholds trigger escalation actions.
    QTLs are predefined limits that, when exceeded, indicate potential systematic issues.
    """
    # Query-related QTLs
    query_aging_warning: int = 15         # Days - amber alert threshold
    query_aging_critical: int = 30        # Days - red alert threshold
    query_rate_per_crf: float = 0.05      # >5% queries per CRF = concern

    # Data completeness QTLs
    missing_page_rate: float = 0.02       # >2% missing pages = concern
    missing_visit_rate: float = 0.01      # >1% missing visits = concern

    # Safety QTLs (stricter thresholds)
    sae_reconciliation_hours: int = 48    # Max hours for SAE reconciliation
    uncoded_terms_threshold: int = 5      # Max uncoded terms before alert

    # SDV QTLs (targeted SDV approach)
    sdv_completion_target: float = 0.80   # 80% SDV for risk-based approach

    # Site performance QTLs
    site_dqi_warning: float = 85.0        # DQI below this = needs attention
    site_dqi_critical: float = 70.0       # DQI below this = escalation


@dataclass
class KRIConfig:
    """
    Key Risk Indicator (KRI) Definitions

    KRIs are specific, quantifiable measures used to track and evaluate
    potential risk exposures within a trial. Per RBQM framework.
    """
    # Data Quality KRIs
    data_entry_error_rate: str = "Non-conformant CRFs / Total CRFs"
    query_resolution_time: str = "Days from query open to close"
    first_pass_resolution_rate: str = "Queries resolved on first response / Total queries"

    # Enrollment KRIs
    enrollment_rate: str = "Subjects enrolled / Site-month"
    screen_failure_rate: str = "Screen failures / Total screened"

    # Safety KRIs
    ae_reporting_timeliness: str = "AEs reported within 24hrs / Total AEs"
    sae_reconciliation_rate: str = "Reconciled SAEs / Total SAEs"

    # Site Performance KRIs
    protocol_deviation_rate: str = "PDs per subject"
    patient_retention_rate: str = "Retained subjects / Enrolled subjects"


# Agent System Prompts
AGENT_SYSTEM_PROMPTS = {
    "orchestrator": """You are the NEXUS CDM Orchestrator Agent - the central coordinator for clinical trial data analysis.

Your role is to:
1. Interpret user queries about clinical trial data
2. Route requests to the appropriate specialist agent(s)
3. Synthesize responses from multiple agents into coherent insights
4. Ensure all outputs are actionable and stakeholder-appropriate

You coordinate three specialist agents:
- Data Steward Agent: Query management, data cleaning, CRF completion
- Medical Monitor Agent: Safety data, SAE reconciliation, protocol deviations
- Regulatory Agent: Submission readiness, audit preparation, compliance

Always use proper clinical trial terminology (RBQM, KRI, QTL, ALCOA, SDV, etc.)
Provide specific, quantified insights with site/patient IDs when relevant.
""",

    "data_steward": """You are the NEXUS CDM Data Steward Agent - an expert Clinical Data Manager.

Your expertise includes:
- Query management and aging analysis (target: <3.2 days resolution)
- Missing page and CRF completion tracking
- Data entry error detection and cleaning recommendations
- CDISC/SDTM compliance verification

When analyzing data:
- Reference specific sites and subjects by ID
- Quantify issues (e.g., "23 queries aging >15 days")
- Prioritize by RBQM risk indicators
- Suggest specific corrective actions

Use proper terminology: EDC, CRF, query aging, data lock, ALCOA principles.
""",

    "medical_monitor": """You are the NEXUS CDM Medical Monitor Agent - a clinical safety specialist.

Your expertise includes:
- SAE reconciliation and safety signal detection
- Protocol deviation tracking and severity assessment
- MedDRA/WHO-Drug coding review
- Adverse event pattern analysis

When analyzing safety data:
- Prioritize patient safety above all else
- Flag any SAE reconciliation gaps immediately
- Assess protocol deviations by severity (major/minor)
- Identify potential safety signals across sites

Use proper terminology: SAE, SUSAR, MedDRA, WHO-Drug, pharmacovigilance.
""",

    "regulatory": """You are the NEXUS CDM Regulatory Agent - a submission and compliance specialist.

Your expertise includes:
- Database lock readiness assessment
- Audit/inspection preparation
- ICH E6(R3) compliance verification
- CDISC submission package requirements

When assessing compliance:
- Reference specific regulatory requirements (ICH E6(R3) sections)
- Provide go/no-go recommendations with justification
- Generate audit-ready documentation
- Identify any blockers to submission

Use proper terminology: database lock, audit trail, 21 CFR Part 11, submission package.
"""
}

# Study metadata
STUDY_NAMES = ["Study 1", "Study 10", "Study 11", "Study 13", "Study 14", "Study 15", "Study 16"]

# File patterns for data loading
FILE_PATTERNS = {
    "edc_metrics": "EDC_Metrics",
    "visit_projection": "Visit",
    "missing_lab": "Lab_Name",
    "sae_dashboard": "SAE",
    "missing_pages": "Missing_Pages",
    "edrr": "EDRR",
    "medra_coding": "MedDRA",
    "whodd_coding": "WHODD",
    "inactivated": "Inactivated"
}


def get_openai_client():
    """Initialize OpenAI client with API key"""
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def create_gpt52_response(input_text, tools=None, reasoning_effort="medium", verbosity="medium", previous_response_id=None):
    """
    Create GPT-5.2 response using the Responses API.

    Args:
        input_text: User input or conversation
        tools: List of tool definitions
        reasoning_effort: "none" | "low" | "medium" | "high" | "xhigh"
        verbosity: "low" | "medium" | "high"
        previous_response_id: For multi-turn conversations

    Returns:
        Response object from GPT-5.2
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    params = {
        "model": OPENAI_MODEL,
        "input": input_text,
        "reasoning": {"effort": reasoning_effort},
        "text": {"verbosity": verbosity}
    }

    if tools:
        params["tools"] = tools

    if previous_response_id:
        params["previous_response_id"] = previous_response_id

    return client.responses.create(**params)


# Initialize default configs
DQI_WEIGHTS = DQIWeightConfig()
QTL_CONFIG = QTLConfig()
KRI_CONFIG = KRIConfig()
