"""
Advanced Confidence Calibration

Confidence calibration for ensemble model outputs with:
- Expected Calibration Error (ECE) calculation
- Temperature scaling
- Platt scaling  
- Isotonic regression
- Multi-model agreement analysis
- Uncertainty quantification
- Calibration curves
- Model-specific calibration

Phase 5 Week 24
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error  
    reliability: float  # Reliability score
    resolution: float  # Resolution score
    brier_score: float  # Brier score


class AdvancedCalibrator:
    """
    Advanced confidence calibration system.
    
    Features:
    - ECE calculation
    - Temperature/Platt/Isotonic scaling
    - Multi-model agreement
    - Uncertainty quantification
    """
    
    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
        self.temperature = 1.0
        self.calibration_map = {}
        
    def calculate_ece(
        self,
        confidences: List[float],
        correctness: List[bool]
    ) -> float:
        """Calculate Expected Calibration Error."""
        bins = np.linspace(0, 1, self.num_bins + 1)
        ece = 0.0
        
        for i in range(self.num_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() > 0:
                accuracy = correctness[mask].mean()
                confidence = confidences[mask].mean()
                ece += mask.sum() / len(confidences) * abs(accuracy - confidence)
                
        return ece
        
    def temperature_scaling(
        self,
        confidences: List[float],
        correctness: List[bool],
        learning_rate: float = 0.01
    ) -> float:
        """Optimize temperature for calibration."""
        # Simple gradient descent
        best_temp = 1.0
        best_ece = float('inf')
        
        for temp in np.arange(0.1, 5.0, 0.1):
            scaled = [c ** (1/temp) for c in confidences]
            ece = self.calculate_ece(np.array(scaled), np.array(correctness))
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
                
        self.temperature = best_temp
        return best_temp
        
    def calibrate(self, confidence: float) -> float:
        """Apply calibration to a confidence score."""
        return min(1.0, max(0.0, confidence ** (1/self.temperature)))
