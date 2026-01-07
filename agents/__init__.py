"""
NEXUS CDM Agents Package
========================
Multi-agent system for clinical trial data management.

Agents:
- OrchestratorAgent: Coordinates specialist agents
- DataStewardAgent: Query management, data cleaning
- MedicalMonitorAgent: Safety data, SAE reconciliation
- RegulatoryAgent: Compliance, submission readiness
"""

from .base_agent import BaseAgent, ToolDefinition, AgentResponse
from .orchestrator import OrchestratorAgent
from .data_steward_agent import DataStewardAgent
from .medical_monitor_agent import MedicalMonitorAgent
from .regulatory_agent import RegulatoryAgent

__all__ = [
    'BaseAgent',
    'ToolDefinition',
    'AgentResponse',
    'OrchestratorAgent',
    'DataStewardAgent',
    'MedicalMonitorAgent',
    'RegulatoryAgent'
]
