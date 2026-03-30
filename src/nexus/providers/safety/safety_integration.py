"""
Unified Safety Pipeline - Complete Integration of All Safety Systems

This module provides a unified safety pipeline that integrates:
- Production Safety System (8 ethical frameworks)
- Enhanced Safety Monitor (5 core rules, compliance scoring)
- Bias Mitigation System (8 bias types, virtue assessment)
- Quarantine System (auto-quarantine on critical violations)

The unified pipeline provides:
- Comprehensive safety evaluation
- Multi-layered protection (rules → bias → ethics)
- Automatic quarantine and rejection decisions
- Detailed safety reports with scores from all systems
- Production-ready safety enforcement

Nexus Integration Phase 2 - Week 3
Based on: NEXUS_INTEGRATION_ROADMAP.md unified safety pipeline design
"""

import logging
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

if TYPE_CHECKING:
    from nexus.providers.safety.production_safety import ProductionSafetySystem
    from nexus.providers.oversight.safety_monitor import SafetyMonitorSystem
    from nexus.providers.safety.bias_mitigation import BiasMitigationSystem

logger = logging.getLogger(__name__)


class SafetyDecision(str, Enum):
    """Final safety decision for content."""
    APPROVE = "approve"
    QUARANTINE = "quarantine"
    REJECT = "reject"


@dataclass
class UnifiedSafetyResult:
    """Complete safety evaluation results from all systems."""
    timestamp: datetime
    decision: SafetyDecision
    reason: str
    safety_score: float  # Overall safety score (0-1)

    # Individual system results
    monitor_result: Dict[str, Any]
    bias_result: Optional[Dict[str, Any]]
    ethics_result: Optional[Dict[str, Any]]

    # Detailed metrics
    compliance_score: float  # From safety monitor (0-100)
    bias_severity: Optional[str]  # low/medium/high/critical
    ethics_risk: Optional[float]  # 0-1

    # Violation details
    violations: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]

    # Quarantine info
    quarantined: bool
    quarantine_id: Optional[str]


class UnifiedSafetyPipeline:
    """
    Unified Safety Pipeline - Complete Safety Integration.

    Integrates all safety systems into a single comprehensive pipeline:
    1. Safety Monitor (5 core rules, compliance scoring)
    2. Bias Detection (8 bias types, severity scoring)
    3. Ethics Assessment (8 ethical frameworks, risk calculation)
    4. Quarantine System (auto-quarantine critical violations)

    Decision Logic:
    - CRITICAL safety rule violation → QUARANTINE (automatic)
    - HIGH bias + HIGH ethics risk → QUARANTINE
    - Ethics risk > 0.8 → REJECT
    - Otherwise → APPROVE (with safety score)

    Example Usage:
        >>> from unified_intelligence.safety import UnifiedSafetyPipeline
        >>>
        >>> pipeline = UnifiedSafetyPipeline()
        >>> await pipeline.initialize()
        >>>
        >>> result = await pipeline.evaluate_output(
        ...     output="AI response text to evaluate",
        ...     context={
        ...         'scenario': 'User query response',
        ...         'stakeholders': ['users', 'public']
        ...     }
        ... )
        >>>
        >>> if result.decision == SafetyDecision.APPROVE:
        ...     print(f"Safe to use (score: {result.safety_score:.2f})")
        >>> elif result.decision == SafetyDecision.QUARANTINE:
        ...     print(f"Quarantined: {result.reason}")
        >>> else:
        ...     print(f"Rejected: {result.reason}")
    """

    def __init__(
        self,
        enable_bias_detection: bool = True,
        enable_ethics_assessment: bool = True,
        enable_monitor: bool = True
    ):
        """
        Initialize unified safety pipeline.

        Args:
            enable_bias_detection: Enable bias mitigation system
            enable_ethics_assessment: Enable ethics assessment
            enable_monitor: Enable safety monitor (core rules)
        """
        self.enable_bias = enable_bias_detection
        self.enable_ethics = enable_ethics_assessment
        self.enable_monitor = enable_monitor

        # Safety systems (lazy loaded)
        self._production_safety: Optional['ProductionSafetySystem'] = None
        self._safety_monitor: Optional['SafetyMonitorSystem'] = None
        self._bias_mitigation: Optional['BiasMitigationSystem'] = None

        self.initialized = False

        # Statistics
        self.total_evaluations = 0
        self.total_approved = 0
        self.total_quarantined = 0
        self.total_rejected = 0

        logger.info("🛡️ Unified Safety Pipeline created")

    async def initialize(self) -> bool:
        """
        Initialize all safety systems.

        Returns:
            True if initialization successful
        """
        try:
            # 1. Initialize Safety Monitor (5 core rules)
            if self.enable_monitor:
                from nexus.providers.oversight.safety_monitor import SafetyMonitorSystem

                self._safety_monitor = SafetyMonitorSystem()
                self._safety_monitor.initialize()
                logger.info("  ✓ Safety Monitor initialized (5 core rules)")

            # 2. Initialize Production Safety (bias + ethics)
            if self.enable_bias or self.enable_ethics:
                from nexus.providers.safety.production_safety import ProductionSafetySystem

                self._production_safety = ProductionSafetySystem()
                logger.info("  ✓ Production Safety initialized (8 frameworks)")

            # 3. Initialize Bias Mitigation (optional, for advanced bias detection)
            if self.enable_bias:
                try:
                    from nexus.providers.safety.bias_mitigation import BiasMitigationSystem

                    self._bias_mitigation = BiasMitigationSystem()
                    logger.info("  ✓ Bias Mitigation initialized (8 bias types)")
                except ImportError:
                    logger.warning("  ⚠ Bias Mitigation not available, using Production Safety only")

            self.initialized = True
            logger.info("✅ Unified Safety Pipeline initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Unified Safety Pipeline: {e}")
            return False

    async def evaluate_output(
        self,
        output: str,
        context: Optional[Dict[str, Any]] = None
    ) -> UnifiedSafetyResult:
        """
        Run output through complete safety pipeline.

        Pipeline stages:
        1. Safety Monitor (5 core rules) - Fast keyword-based checks
        2. Bias Detection (if enabled) - Pattern and AI-based bias
        3. Ethics Assessment (if enabled) - Multi-framework evaluation
        4. Decision logic - Approve, Quarantine, or Reject

        Args:
            output: Content to evaluate
            context: Optional context (scenario, stakeholders, etc.)

        Returns:
            UnifiedSafetyResult with decision and detailed results
        """
        if not self.initialized:
            await self.initialize()

        self.total_evaluations += 1
        context = context or {}

        results = {
            'timestamp': datetime.now(),
            'output': output,
            'context': context
        }

        violations: List[Dict[str, Any]] = []
        alerts: List[Dict[str, Any]] = []

        # ═══════════════════════════════════════════════════════
        # STAGE 1: Enhanced Safety Monitor (5 core rules)
        # ═══════════════════════════════════════════════════════
        monitor_result = None
        compliance_score = 100.0

        if self._safety_monitor:
            # Prepare experience data for monitor
            experience_data = {
                'content': output,
                'description': context.get('scenario', output[:200])
            }

            safety_eval = self._safety_monitor.evaluate_safety(
                experience_data,
                context
            )

            monitor_result = {
                'safe': safety_eval.safe,
                'violations': [asdict(v) for v in safety_eval.violations],
                'alerts': safety_eval.alerts,
                'quarantine_required': safety_eval.quarantine_required,
                'compliance_score': safety_eval.compliance_score
            }

            compliance_score = safety_eval.compliance_score
            violations.extend([asdict(v) for v in safety_eval.violations])
            alerts.extend(safety_eval.alerts)

            results['monitor'] = monitor_result

            # Auto-quarantine on critical safety rule violation
            if safety_eval.quarantine_required:
                critical_violation = next(
                    (v for v in safety_eval.violations
                     if v.severity.value == 'critical'),
                    None
                )

                quarantine_id = f"quarantine_{int(datetime.now().timestamp() * 1000)}"

                result = UnifiedSafetyResult(
                    timestamp=datetime.now(),
                    decision=SafetyDecision.QUARANTINE,
                    reason=f"Critical safety violation: {critical_violation.rule_name if critical_violation else 'Unknown'}",
                    safety_score=0.0,
                    monitor_result=monitor_result,
                    bias_result=None,
                    ethics_result=None,
                    compliance_score=compliance_score,
                    bias_severity=None,
                    ethics_risk=None,
                    violations=violations,
                    alerts=alerts,
                    quarantined=True,
                    quarantine_id=quarantine_id
                )

                self.total_quarantined += 1
                logger.warning(f"⚠️ QUARANTINED: {result.reason}")

                return result

        # ═══════════════════════════════════════════════════════
        # STAGE 2: Bias Detection
        # ═══════════════════════════════════════════════════════
        bias_result = None
        bias_severity = None

        if self._production_safety and self.enable_bias:
            bias_detection = self._production_safety.detect_bias(
                output,
                context.get('description', context.get('scenario'))
            )

            bias_result = {
                'bias_type': bias_detection.bias_type,
                'severity': bias_detection.severity,
                'confidence': bias_detection.confidence,
                'mitigation_suggestions': bias_detection.mitigation_suggestions
            }

            bias_severity = bias_detection.severity
            results['bias'] = bias_result

            # Add to violations if high/critical bias
            if bias_severity in ['high', 'critical']:
                violations.append({
                    'type': 'bias',
                    'severity': bias_severity,
                    'details': bias_result
                })

        # ═══════════════════════════════════════════════════════
        # STAGE 3: Ethics Assessment
        # ═══════════════════════════════════════════════════════
        ethics_result = None
        ethics_risk = 0.0

        if self._production_safety and self.enable_ethics:
            ethics_assessment = await self._production_safety.assess_ethics(
                scenario=context.get('scenario', output[:200]),
                stakeholders=context.get('stakeholders', ['users'])
            )

            ethics_result = {
                'ethical_frameworks': ethics_assessment.ethical_frameworks,
                'risk_level': ethics_assessment.risk_level,
                'concerns': ethics_assessment.concerns,
                'required_actions': ethics_assessment.required_actions
            }

            ethics_risk = ethics_assessment.risk_level
            results['ethics'] = ethics_result

            # Add to violations if high ethics risk
            if ethics_risk > 0.7:
                violations.append({
                    'type': 'ethics',
                    'severity': 'high' if ethics_risk > 0.8 else 'medium',
                    'risk_level': ethics_risk,
                    'details': ethics_result
                })

        # ═══════════════════════════════════════════════════════
        # STAGE 4: Decision Logic
        # ═══════════════════════════════════════════════════════
        decision, reason, quarantined, quarantine_id = self._make_safety_decision(
            monitor_result,
            bias_result,
            bias_severity,
            ethics_result,
            ethics_risk
        )

        # Calculate overall safety score
        safety_score = self._calculate_safety_score(
            compliance_score,
            bias_severity,
            ethics_risk
        )

        # Create final result
        result = UnifiedSafetyResult(
            timestamp=datetime.now(),
            decision=decision,
            reason=reason,
            safety_score=safety_score,
            monitor_result=monitor_result or {},
            bias_result=bias_result,
            ethics_result=ethics_result,
            compliance_score=compliance_score,
            bias_severity=bias_severity,
            ethics_risk=ethics_risk,
            violations=violations,
            alerts=alerts,
            quarantined=quarantined,
            quarantine_id=quarantine_id
        )

        # Update statistics
        if decision == SafetyDecision.APPROVE:
            self.total_approved += 1
        elif decision == SafetyDecision.QUARANTINE:
            self.total_quarantined += 1
        elif decision == SafetyDecision.REJECT:
            self.total_rejected += 1

        # Log decision
        if decision == SafetyDecision.APPROVE:
            logger.info(f"✅ APPROVED (score: {safety_score:.2f})")
        elif decision == SafetyDecision.QUARANTINE:
            logger.warning(f"⚠️ QUARANTINED: {reason}")
        else:
            logger.error(f"❌ REJECTED: {reason}")

        return result

    def _make_safety_decision(
        self,
        monitor_result: Optional[Dict],
        bias_result: Optional[Dict],
        bias_severity: Optional[str],
        ethics_result: Optional[Dict],
        ethics_risk: Optional[float]
    ) -> tuple[SafetyDecision, str, bool, Optional[str]]:
        """
        Make final safety decision based on all results.

        Decision logic:
        1. HIGH bias + HIGH ethics risk → QUARANTINE
        2. Ethics risk > 0.8 → REJECT
        3. Monitor violations (non-critical) → APPROVE with warning
        4. Otherwise → APPROVE

        Returns:
            (decision, reason, quarantined, quarantine_id)
        """
        # Case 1: Combined high bias + high ethics risk
        if bias_severity in ['high', 'critical'] and ethics_risk and ethics_risk > 0.7:
            quarantine_id = f"quarantine_{int(datetime.now().timestamp() * 1000)}"
            return (
                SafetyDecision.QUARANTINE,
                f"High bias ({bias_result['bias_type']}) + High ethics risk ({ethics_risk:.2f})",
                True,
                quarantine_id
            )

        # Case 2: Unacceptable ethics risk alone
        if ethics_risk and ethics_risk > 0.8:
            return (
                SafetyDecision.REJECT,
                f"Unacceptable ethics risk: {ethics_risk:.2f}",
                False,
                None
            )

        # Case 3: Critical bias alone (edge case, usually caught earlier)
        if bias_severity == 'critical':
            quarantine_id = f"quarantine_{int(datetime.now().timestamp() * 1000)}"
            return (
                SafetyDecision.QUARANTINE,
                f"Critical bias detected: {bias_result['bias_type'] if bias_result else 'Unknown'}",
                True,
                quarantine_id
            )

        # Case 4: High ethics risk (warning level)
        if ethics_risk and ethics_risk > 0.6:
            return (
                SafetyDecision.APPROVE,
                f"Approved with ethics warning (risk: {ethics_risk:.2f})",
                False,
                None
            )

        # Case 5: Medium/high bias (warning level)
        if bias_severity in ['medium', 'high']:
            return (
                SafetyDecision.APPROVE,
                f"Approved with bias warning ({bias_severity} {bias_result['bias_type'] if bias_result else ''})",
                False,
                None
            )

        # Case 6: All clear
        return (
            SafetyDecision.APPROVE,
            "All safety checks passed",
            False,
            None
        )

    def _calculate_safety_score(
        self,
        compliance_score: float,
        bias_severity: Optional[str],
        ethics_risk: Optional[float]
    ) -> float:
        """
        Calculate overall safety score (0-1).

        Weighted combination:
        - Compliance: 40% (from safety monitor 0-100 → 0-1)
        - Bias: 30% (inverted severity)
        - Ethics: 30% (inverted risk)
        """
        # Compliance component (0-100 → 0-1)
        compliance_component = (compliance_score / 100.0) * 0.4

        # Bias component (inverted severity)
        bias_severity_map = {
            'critical': 0.0,
            'high': 0.3,
            'medium': 0.6,
            'low': 0.9,
            None: 1.0
        }
        bias_component = bias_severity_map.get(bias_severity, 1.0) * 0.3

        # Ethics component (inverted risk)
        ethics_component = (1.0 - (ethics_risk or 0.0)) * 0.3

        # Combined score
        safety_score = compliance_component + bias_component + ethics_component

        return min(1.0, max(0.0, safety_score))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with evaluation statistics
        """
        return {
            'total_evaluations': self.total_evaluations,
            'total_approved': self.total_approved,
            'total_quarantined': self.total_quarantined,
            'total_rejected': self.total_rejected,
            'approval_rate': (
                self.total_approved / self.total_evaluations
                if self.total_evaluations > 0 else 0.0
            ),
            'quarantine_rate': (
                self.total_quarantined / self.total_evaluations
                if self.total_evaluations > 0 else 0.0
            ),
            'rejection_rate': (
                self.total_rejected / self.total_evaluations
                if self.total_evaluations > 0 else 0.0
            ),
            'systems_enabled': {
                'monitor': self.enable_monitor,
                'bias_detection': self.enable_bias,
                'ethics_assessment': self.enable_ethics
            }
        }

    def get_compliance_score(self) -> float:
        """
        Get current compliance score from safety monitor.

        Returns:
            Compliance score (0-100)
        """
        if self._safety_monitor:
            return self._safety_monitor.get_compliance_score()
        return 100.0

    def get_quarantine_queue(self) -> List[Dict[str, Any]]:
        """
        Get current quarantine queue from safety monitor.

        Returns:
            List of quarantined items
        """
        if self._safety_monitor:
            items = self._safety_monitor.get_quarantine_queue()
            return [asdict(item) for item in items]
        return []

    def clear_quarantine(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear quarantine queue.

        Args:
            older_than_days: Only clear items older than this many days

        Returns:
            Number of items cleared
        """
        if self._safety_monitor:
            return self._safety_monitor.clear_quarantine(older_than_days)
        return 0
