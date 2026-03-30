"""
Registry of csuite executives and their capabilities.

Maps task types to the appropriate executive codes.
Used by AutonomousCOO to route work through csuite.

This registry enables Nexus to understand what capabilities each
csuite executive has and route tasks accordingly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class ExecutiveInfo:
    """Information about a csuite executive."""
    code: str
    name: str
    codename: str
    domain: str
    capabilities: List[str] = field(default_factory=list)
    description: str = ""


# Complete registry of all 16 csuite executives
EXECUTIVE_CAPABILITIES: Dict[str, ExecutiveInfo] = {
    "CTO": ExecutiveInfo(
        code="CTO",
        name="Chief Technology Officer",
        codename="Forge",
        domain="Development, Engineering, DevOps",
        capabilities=[
            "code_review",
            "code_generation",
            "architecture",
            "deployment",
            "testing",
            "debugging",
            "refactoring",
            "technical_design",
        ],
        description="Leads technical strategy, code quality, and software architecture.",
    ),
    "CMO": ExecutiveInfo(
        code="CMO",
        name="Chief Marketing Officer",
        codename="Echo",
        domain="Marketing, Content, Brand",
        capabilities=[
            "campaign_creation",
            "content_strategy",
            "brand_analysis",
            "social_distribute",
            "social_publish",
            "social_schedule",
            "social_analytics",
            "social_metrics",
        ],
        description="Drives marketing strategy, content creation, and brand development.",
    ),
    "CFO": ExecutiveInfo(
        code="CFO",
        name="Chief Financial Officer",
        codename="Keystone",
        domain="Finance, Budgeting, Cost Management",
        capabilities=[
            "budget_analysis",
            "cost_optimization",
            "financial_report",
            "roi_analysis",
        ],
        description="Manages financial planning, budgeting, and cost optimization.",
    ),
    "CIO": ExecutiveInfo(
        code="CIO",
        name="Chief Information Officer",
        codename="Sentinel",
        domain="Information Security, Access Control",
        capabilities=[
            "security_scan",
            "security_review",
            "threat_analysis",
            "access_control",
        ],
        description="Oversees information security and threat detection.",
    ),
    "CSecO": ExecutiveInfo(
        code="CSecO",
        name="Chief Security Officer",
        codename="Citadel",
        domain="Security Operations, Incident Response",
        capabilities=[
            "vulnerability_check",
            "incident_response",
            "security_audit",
            "penetration_testing",
        ],
        description="Leads security operations and incident response.",
    ),
    "CSO": ExecutiveInfo(
        code="CSO",
        name="Chief Strategy Officer",
        codename="Compass",
        domain="Strategy, Market Analysis, Competition",
        capabilities=[
            "strategic_analysis",
            "market_research",
            "competitive_analysis",
            "strategic_planning",
        ],
        description="Develops strategic direction and market positioning.",
    ),
    "CRO": ExecutiveInfo(
        code="CRO",
        name="Chief Research Officer",
        codename="Axiom",
        domain="Research, Data Analysis, Insights",
        capabilities=[
            "research",
            "data_analysis",
            "insights",
            "trend_analysis",
        ],
        description="Leads research initiatives and data-driven insights.",
    ),
    "CDO": ExecutiveInfo(
        code="CDO",
        name="Chief Data Officer",
        codename="Index",
        domain="Data Governance, Knowledge Management",
        capabilities=[
            "data_governance",
            "data_quality",
            "knowledge_management",
            "data_architecture",
        ],
        description="Manages data governance and knowledge systems.",
    ),
    "CComO": ExecutiveInfo(
        code="CComO",
        name="Chief Compliance Officer",
        codename="Accord",
        domain="Compliance, Audit, Policy",
        capabilities=[
            "compliance_check",
            "audit",
            "policy_review",
            "regulatory_analysis",
        ],
        description="Ensures compliance and policy adherence.",
    ),
    "CRiO": ExecutiveInfo(
        code="CRiO",
        name="Chief Risk Officer",
        codename="Aegis",
        domain="Risk Management, Mitigation",
        capabilities=[
            "risk_assessment",
            "risk_mitigation",
            "risk_monitoring",
            "risk_reporting",
        ],
        description="Identifies and mitigates organizational risks.",
    ),
    "CPO": ExecutiveInfo(
        code="CPO",
        name="Chief Product Officer",
        codename="Blueprint",
        domain="Product Strategy, Feature Planning",
        capabilities=[
            "product_strategy",
            "feature_planning",
            "user_research",
            "product_roadmap",
        ],
        description="Drives product vision and roadmap development.",
    ),
    "CCO": ExecutiveInfo(
        code="CCO",
        name="Chief Customer Officer",
        codename="Beacon",
        domain="Customer Success, Feedback",
        capabilities=[
            "customer_feedback",
            "customer_success",
            "customer_satisfaction",
            "customer_journey",
        ],
        description="Champions customer experience and satisfaction.",
    ),
    "CRevO": ExecutiveInfo(
        code="CRevO",
        name="Chief Revenue Officer",
        codename="Vector",
        domain="Revenue, Sales Strategy",
        capabilities=[
            "revenue_optimization",
            "sales_strategy",
            "revenue_summary",
            "revenue_tracking",
            "pipeline_management",
        ],
        description="Maximizes revenue and sales performance.",
    ),
    "CEngO": ExecutiveInfo(
        code="CEngO",
        name="Chief Engineering Officer",
        codename="Foundry",
        domain="Infrastructure, Platform Engineering",
        capabilities=[
            "infrastructure",
            "devops",
            "platform_engineering",
            "system_architecture",
        ],
        description="Leads infrastructure and platform development.",
    ),
    "CoS": ExecutiveInfo(
        code="CoS",
        name="Chief of Staff",
        codename="Overwatch",
        domain="Operations, Coordination, Routing",
        capabilities=[
            "task_routing",
            "coordination",
            "operations_management",
            "executive_support",
        ],
        description="Coordinates operations and routes tasks to executives.",
    ),
    "COO": ExecutiveInfo(
        code="COO",
        name="Chief Operating Officer",
        codename="Nexus",
        domain="Operations, Strategic Execution",
        capabilities=[
            "operations_management",
            "strategic_execution",
            "process_optimization",
        ],
        description="Oversees day-to-day operations (deprecated, use CoS).",
    ),
}


# Build reverse mapping: capability/task_type -> executive code
_TASK_TO_EXECUTIVE: Dict[str, str] = {}

for exec_code, exec_info in EXECUTIVE_CAPABILITIES.items():
    for capability in exec_info.capabilities:
        # First executive to claim a capability wins
        if capability not in _TASK_TO_EXECUTIVE:
            _TASK_TO_EXECUTIVE[capability] = exec_code


# Additional keyword-based routing for natural language matching
KEYWORD_ROUTING: Dict[str, str] = {
    # Technical keywords -> CTO
    "code": "CTO",
    "programming": "CTO",
    "software": "CTO",
    "development": "CTO",
    "implement": "CTO",
    "build": "CTO",
    "fix": "CTO",
    "debug": "CTO",
    "test": "CTO",
    "api": "CTO",
    # Security keywords -> CIO/CSecO
    "security": "CIO",
    "vulnerability": "CSecO",
    "threat": "CIO",
    "attack": "CSecO",
    "breach": "CSecO",
    "firewall": "CIO",
    "encryption": "CIO",
    # Marketing keywords -> CMO
    "marketing": "CMO",
    "campaign": "CMO",
    "brand": "CMO",
    "social media": "CMO",
    "content": "CMO",
    "advertising": "CMO",
    "promotion": "CMO",
    # Financial keywords -> CFO
    "budget": "CFO",
    "financial": "CFO",
    "cost": "CFO",
    "expense": "CFO",
    "roi": "CFO",
    "profit": "CFO",
    "revenue forecast": "CFO",
    # Strategy keywords -> CSO
    "strategy": "CSO",
    "strategic": "CSO",
    "competitive": "CSO",
    "market analysis": "CSO",
    "positioning": "CSO",
    # Research keywords -> CRO
    "research": "CRO",
    "study": "CRO",
    "analyze": "CRO",
    "investigate": "CRO",
    "discover": "CRO",
    # Data keywords -> CDO
    "data governance": "CDO",
    "data quality": "CDO",
    "knowledge base": "CDO",
    "metadata": "CDO",
    # Compliance keywords -> CComO
    "compliance": "CComO",
    "audit": "CComO",
    "policy": "CComO",
    "regulation": "CComO",
    "legal": "CComO",
    # Risk keywords -> CRiO
    "risk": "CRiO",
    "mitigation": "CRiO",
    "exposure": "CRiO",
    # Product keywords -> CPO
    "product": "CPO",
    "feature": "CPO",
    "roadmap": "CPO",
    "user experience": "CPO",
    "ux": "CPO",
    # Customer keywords -> CCO
    "customer": "CCO",
    "client": "CCO",
    "support": "CCO",
    "satisfaction": "CCO",
    "feedback": "CCO",
    # Revenue keywords -> CRevO
    "sales": "CRevO",
    "revenue": "CRevO",
    "pipeline": "CRevO",
    "deal": "CRevO",
    "conversion": "CRevO",
    # Engineering keywords -> CEngO
    "infrastructure": "CEngO",
    "devops": "CEngO",
    "deployment": "CEngO",
    "platform": "CEngO",
    "scalability": "CEngO",
    "monitoring": "CEngO",
}


def get_executive_for_task(task_type: str) -> Optional[str]:
    """
    Find the best executive for a task type.

    Args:
        task_type: The type of task (e.g., "code_review", "campaign_creation")

    Returns:
        Executive code (e.g., "CTO", "CMO") or None if no match found
    """
    # Normalize task type
    normalized = task_type.lower().strip()

    # Direct mapping from task type
    if normalized in _TASK_TO_EXECUTIVE:
        return _TASK_TO_EXECUTIVE[normalized]

    # Try with underscores replaced by spaces
    normalized_spaced = normalized.replace("_", " ")
    if normalized_spaced in _TASK_TO_EXECUTIVE:
        return _TASK_TO_EXECUTIVE[normalized_spaced]

    return None


def get_executive_for_text(text: str) -> Optional[str]:
    """
    Find the best executive based on text content using keyword matching.

    Args:
        text: Text to analyze (e.g., title, description)

    Returns:
        Executive code (e.g., "CTO", "CMO") or None if no match found
    """
    if not text:
        return None

    text_lower = text.lower()

    # Check keyword routing (longer keywords first for specificity)
    sorted_keywords = sorted(KEYWORD_ROUTING.keys(), key=len, reverse=True)

    for keyword in sorted_keywords:
        if keyword in text_lower:
            return KEYWORD_ROUTING[keyword]

    return None


def get_executive_info(code: str) -> Optional[ExecutiveInfo]:
    """
    Get information about an executive.

    Args:
        code: Executive code (e.g., "CTO", "CMO")

    Returns:
        ExecutiveInfo or None if not found
    """
    return EXECUTIVE_CAPABILITIES.get(code.upper())


def get_all_executives() -> List[ExecutiveInfo]:
    """
    Get list of all executives.

    Returns:
        List of ExecutiveInfo for all executives
    """
    return list(EXECUTIVE_CAPABILITIES.values())


def get_executive_codes() -> List[str]:
    """
    Get list of all executive codes.

    Returns:
        List of codes (e.g., ["CTO", "CMO", "CFO", ...])
    """
    return list(EXECUTIVE_CAPABILITIES.keys())


def get_capabilities_for_executive(code: str) -> List[str]:
    """
    Get capabilities for an executive.

    Args:
        code: Executive code

    Returns:
        List of capabilities or empty list if not found
    """
    exec_info = EXECUTIVE_CAPABILITIES.get(code.upper())
    if exec_info:
        return exec_info.capabilities.copy()
    return []


def get_all_task_types() -> Set[str]:
    """
    Get all known task types.

    Returns:
        Set of all task types that can be routed
    """
    return set(_TASK_TO_EXECUTIVE.keys())


def find_executives_for_capability(capability: str) -> List[str]:
    """
    Find all executives that have a specific capability.

    Args:
        capability: Capability to search for

    Returns:
        List of executive codes that have this capability
    """
    capability_lower = capability.lower()
    matches = []

    for exec_code, exec_info in EXECUTIVE_CAPABILITIES.items():
        for cap in exec_info.capabilities:
            if capability_lower in cap.lower() or cap.lower() in capability_lower:
                matches.append(exec_code)
                break

    return matches
