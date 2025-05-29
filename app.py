"""
Enhanced Skin Tone Analysis Flask Application
Improved with better fuzzy logic, dynamic recommendations, and personalized explanations
"""

import cv2
import os
import numpy as np
import logging
import uuid
import time
import colorsys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from flask import Flask, render_template, request, url_for, flash, redirect, jsonify
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
MAX_IMAGE_DIMENSION = 1000

@dataclass
class SkinAnalysisResult:
    """Enhanced data class for skin analysis results"""
    overtone: Tuple[int, int, int]
    rgb_color: str
    hex_color: str
    undertone: str
    confidence: float
    confidence_level: str
    fuzzy_scores: Dict[str, float]
    dominant_factors: Dict[str, Any]
    context: Dict[str, Any]
    cropped_image: str
    statistics: Dict[str, float]
    personalized_reason: str

class Config:
    """Application configuration"""
    SECRET_KEY = os.urandom(24)
    UPLOAD_FOLDER = 'static/uploads'
    CROPPED_FOLDER = 'static/uploads/cropped'
    ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS
    MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH

class EnhancedFuzzyUndertoneAnalyzer:
    """Enhanced Fuzzy Logic system with calibrated parameters and advanced analysis"""
    
    def __init__(self):
        self._init_calibrated_params()
        self._init_factor_weights()
        self._init_explanation_templates()
    
    def _init_calibrated_params(self):
        """Initialize calibrated fuzzy membership parameters based on dataset analysis"""
        # Calibrated RB difference parameters (Red - Blue)
        self.rb_diff_params = {
            'cool': {'low': -80, 'mid': -35, 'high': -5},     # Strong blue dominance
            'neutral': {'low': -20, 'mid': 0, 'high': 20},    # Balanced red-blue
            'warm': {'low': 5, 'mid': 30, 'high': 85}         # Strong red dominance
        }
        
        # Calibrated RG ratio parameters (Red / Green)
        self.rg_ratio_params = {
            'cool': {'low': 0.65, 'mid': 0.85, 'high': 0.98},   # Less red vs green
            'neutral': {'low': 0.90, 'mid': 1.0, 'high': 1.10}, # Balanced
            'warm': {'low': 1.02, 'mid': 1.25, 'high': 1.65}    # More red vs green
        }
        
        # Calibrated GB ratio parameters (Green / Blue)
        self.gb_ratio_params = {
            'cool': {'low': 0.75, 'mid': 0.95, 'high': 1.10},   # Balanced/blue dominant
            'neutral': {'low': 0.90, 'mid': 1.05, 'high': 1.20},
            'warm': {'low': 1.15, 'mid': 1.40, 'high': 1.80}    # Green dominant
        }
        
        # HSV-based parameters for advanced analysis
        self.hue_params = {
            'cool': {'low': 180, 'mid': 220, 'high': 280},     # Blue-purple range
            'neutral': {'low': 20, 'mid': 35, 'high': 50},     # Yellow-orange range
            'warm': {'low': 0, 'mid': 15, 'high': 30}          # Red-orange range
        }
        
        # Saturation sensitivity parameters
        self.saturation_params = {
            'cool': {'threshold': 0.15, 'sensitivity': 1.2},
            'neutral': {'threshold': 0.20, 'sensitivity': 1.0},
            'warm': {'threshold': 0.25, 'sensitivity': 0.8}
        }
    
    def _init_factor_weights(self):
        """Initialize dynamic factor weights for rule-based inference"""
        self.base_weights = {
            'rb_diff': 0.30,        # Red-Blue difference (primary indicator)
            'rg_ratio': 0.25,       # Red-Green ratio
            'gb_ratio': 0.20,       # Green-Blue ratio
            'hue_factor': 0.15,     # HSV hue analysis
            'saturation_factor': 0.10  # Saturation-based adjustment
        }
        
        # Dynamic weight adjustments based on dominant factors
        self.weight_adjustments = {
            'high_saturation': {'saturation_factor': 0.20, 'rb_diff': 0.25},
            'low_saturation': {'hue_factor': 0.10, 'rb_diff': 0.35},
            'extreme_rb_diff': {'rb_diff': 0.40, 'rg_ratio': 0.20},
            'balanced_ratios': {'hue_factor': 0.25, 'saturation_factor': 0.15}
        }
    
    def _init_explanation_templates(self):
        """Initialize personalized explanation templates"""
        self.explanation_templates = {
            'cool': {
                'rb_dominant': "Kulit Anda memiliki undertone cool karena dominasi warna biru (RB difference: {rb_diff:.1f}). Ini menciptakan base yang sejuk dan segar.",
                'hue_dominant': "Analisis hue menunjukkan undertone cool dengan nilai {hue_angle:.1f}°. Kulit Anda cocok dengan warna-warna dingin.",
                'saturation_dominant': "Tingkat saturasi {saturation:.2f} pada kulit Anda mendukung undertone cool, memberikan kesan natural dan elegan.",
                'balanced': "Kombinasi seimbang dari faktor-faktor analisis menunjukkan undertone cool yang stabil dan konsisten."
            },
            'warm': {
                'rb_dominant': "Undertone warm teridentifikasi dari dominasi warna merah (RB difference: {rb_diff:.1f}). Ini memberikan kehangatan natural pada kulit.",
                'hue_dominant': "Nilai hue {hue_angle:.1f}° menunjukkan undertone warm yang kuat. Kulit Anda akan bersinar dengan warna-warna hangat.",
                'saturation_dominant': "Saturasi {saturation:.2f} pada kulit mendukung undertone warm, menciptakan glow yang natural dan berseri.",
                'balanced': "Analisis komprehensif menunjukkan undertone warm yang konsisten di berbagai faktor pengukuran."
            },
            'neutral': {
                'balanced_ratios': "Kulit Anda memiliki undertone neutral dengan keseimbangan optimal antara elemen hangat dan dingin (RG ratio: {rg_ratio:.2f}).",
                'mixed_signals': "Kombinasi faktor yang seimbang (RB: {rb_diff:.1f}, Hue: {hue_angle:.1f}°) menunjukkan undertone neutral yang fleksibel.",
                'low_confidence': "Undertone neutral teridentifikasi dengan berbagai indikator yang saling melengkapi, memberikan fleksibilitas dalam pemilihan warna.",
                'versatile': "Analisis menunjukkan undertone neutral yang versatile, cocok dengan spektrum warna yang luas."
            }
        }
    
    @staticmethod
    def triangular_membership(x: float, low: float, mid: float, high: float) -> float:
        """Enhanced triangular membership function with smoothing"""
        if x <= low or x >= high:
            return 0.0
        elif x == mid:
            return 1.0
        elif x < mid:
            return max(0.0, (x - low) / (mid - low))
        else:
            return max(0.0, (high - x) / (high - mid))
    
    def _calculate_hsv_factors(self, r: int, g: int, b: int) -> Dict[str, float]:
        """Calculate HSV-based factors for enhanced analysis"""
        # Convert to HSV
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        hue_deg = h * 360
        
        # Calculate hue-based memberships
        hue_memberships = {}
        for tone in ['cool', 'neutral', 'warm']:
            params = self.hue_params[tone]
            
            # Handle hue circularity
            if tone == 'warm' and hue_deg > 330:  # Red hue wraps around
                hue_deg_adjusted = hue_deg - 360
            else:
                hue_deg_adjusted = hue_deg
            
            hue_memberships[tone] = self.triangular_membership(
                hue_deg_adjusted, params['low'], params['mid'], params['high']
            )
        
        return {
            'hue_angle': hue_deg,
            'saturation': s,
            'value': v,
            'hue_memberships': hue_memberships
        }
    
    def _calculate_advanced_factors(self, r: int, g: int, b: int) -> Dict[str, Any]:
        """Calculate advanced color analysis factors"""
        # Basic ratios
        max_val = max(r, g, b, 1)  # Prevent division by zero
        min_val = min(r, g, b)
        
        # Color dominance
        color_dominance = {
            'red': r / max_val,
            'green': g / max_val,
            'blue': b / max_val
        }
        
        # Color contrast and harmony
        contrast = (max_val - min_val) / max_val
        harmony_score = 1.0 - abs(r - g) / max_val - abs(g - b) / max_val - abs(r - b) / max_val
        
        # Chromatic intensity
        chromatic_intensity = np.sqrt((r - g)**2 + (g - b)**2 + (r - b)**2) / (255 * np.sqrt(2))
        
        return {
            'color_dominance': color_dominance,
            'contrast': contrast,
            'harmony_score': max(0.0, harmony_score),
            'chromatic_intensity': chromatic_intensity
        }
    
    def _extract_dominant_factors(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dominant factors from analysis for dynamic recommendations"""
        factors = {}
        
        # RB difference dominance
        rb_diff = abs(analysis_data['rb_diff'])
        if rb_diff > 40:
            factors['primary'] = 'rb_dominant'
            factors['strength'] = 'high'
        elif rb_diff > 20:
            factors['primary'] = 'rb_moderate'
            factors['strength'] = 'medium'
        
        # Saturation dominance
        saturation = analysis_data['hsv_factors']['saturation']
        if saturation > 0.3:
            factors['secondary'] = 'high_saturation'
        elif saturation < 0.15:
            factors['secondary'] = 'low_saturation'
        
        # Hue clarity
        hue_scores = analysis_data['hsv_factors']['hue_memberships']
        max_hue_score = max(hue_scores.values())
        if max_hue_score > 0.7:
            factors['tertiary'] = 'hue_dominant'
        
        # Ratio balance
        rg_ratio = analysis_data['rg_ratio']
        gb_ratio = analysis_data['gb_ratio']
        if 0.95 <= rg_ratio <= 1.05 and 0.95 <= gb_ratio <= 1.05:
            factors['balance'] = 'highly_balanced'
        elif 0.9 <= rg_ratio <= 1.1 and 0.9 <= gb_ratio <= 1.1:
            factors['balance'] = 'moderately_balanced'
        
        return factors
    
    def _get_dynamic_weights(self, dominant_factors: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dynamic weights based on dominant factors"""
        weights = self.base_weights.copy()
        
        # Adjust weights based on dominant factors
        if dominant_factors.get('primary') == 'rb_dominant':
            weights.update(self.weight_adjustments['extreme_rb_diff'])
        elif dominant_factors.get('secondary') == 'high_saturation':
            weights.update(self.weight_adjustments['high_saturation'])
        elif dominant_factors.get('secondary') == 'low_saturation':
            weights.update(self.weight_adjustments['low_saturation'])
        elif dominant_factors.get('balance') == 'highly_balanced':
            weights.update(self.weight_adjustments['balanced_ratios'])
        
        # Normalize weights
        total_weight = sum(weights.values())
        return {k: v/total_weight for k, v in weights.items()}
    
    def enhanced_fuzzy_inference(self, r_avg: int, g_avg: int, b_avg: int) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Enhanced fuzzy inference with dynamic weighting and advanced factors"""
        # Calculate basic factors
        rb_diff = r_avg - b_avg
        rg_ratio = r_avg / max(g_avg, 1)
        gb_ratio = g_avg / max(b_avg, 1)
        
        # Calculate advanced factors
        hsv_factors = self._calculate_hsv_factors(r_avg, g_avg, b_avg)
        advanced_factors = self._calculate_advanced_factors(r_avg, g_avg, b_avg)
        
        # Compile analysis data
        analysis_data = {
            'rb_diff': rb_diff,
            'rg_ratio': rg_ratio,
            'gb_ratio': gb_ratio,
            'hsv_factors': hsv_factors,
            'advanced_factors': advanced_factors
        }
        
        # Extract dominant factors
        dominant_factors = self._extract_dominant_factors(analysis_data)
        
        # Get dynamic weights
        dynamic_weights = self._get_dynamic_weights(dominant_factors)
        
        logger.info(f"Enhanced analysis - R:{r_avg}, G:{g_avg}, B:{b_avg}")
        logger.info(f"Dominant factors: {dominant_factors}")
        logger.info(f"Dynamic weights: {dynamic_weights}")
        
        # Calculate membership scores for each undertone
        final_scores = {}
        detailed_scores = {}
        
        for tone in ['cool', 'neutral', 'warm']:
            # Basic factor memberships
            rb_membership = self._calculate_membership(rb_diff, self.rb_diff_params, tone)
            rg_membership = self._calculate_membership(rg_ratio, self.rg_ratio_params, tone)
            gb_membership = self._calculate_membership(gb_ratio, self.gb_ratio_params, tone)
            
            # Advanced factor memberships
            hue_membership = hsv_factors['hue_memberships'][tone]
            saturation_membership = self._get_saturation_membership(tone, hsv_factors['saturation'])
            
            # Apply dynamic weights
            weighted_score = (
                rb_membership * dynamic_weights['rb_diff'] +
                rg_membership * dynamic_weights['rg_ratio'] +
                gb_membership * dynamic_weights['gb_ratio'] +
                hue_membership * dynamic_weights['hue_factor'] +
                saturation_membership * dynamic_weights['saturation_factor']
            )
            
            final_scores[tone] = weighted_score
            detailed_scores[tone] = {
                'rb': rb_membership,
                'rg': rg_membership,
                'gb': gb_membership,
                'hue': hue_membership,
                'saturation': saturation_membership,
                'weighted': weighted_score
            }
            
            logger.info(f"{tone}: RB={rb_membership:.3f}, RG={rg_membership:.3f}, "
                       f"GB={gb_membership:.3f}, Hue={hue_membership:.3f}, "
                       f"Sat={saturation_membership:.3f}, Final={weighted_score:.3f}")
        
        # Normalize scores
        self._normalize_scores(final_scores)
        
        # Build comprehensive context
        context = {
            **analysis_data,
            'dominant_factors': dominant_factors,
            'dynamic_weights': dynamic_weights,
            'detailed_scores': detailed_scores,
            'brightness': (r_avg + g_avg + b_avg) / 3 / 255
        }
        
        return final_scores, context
    
    def _calculate_membership(self, value: float, params_dict: Dict, tone: str) -> float:
        """Enhanced membership calculation with boundary handling"""
        params = params_dict[tone]
        membership = self.triangular_membership(value, params['low'], params['mid'], params['high'])
        
        # Apply smoothing for edge cases
        if 0 < membership < 0.1:
            membership *= 0.5  # Reduce very low memberships
        elif membership > 0.9:
            membership = min(1.0, membership * 1.05)  # Slight boost for high memberships
        
        return membership
    
    def _get_saturation_membership(self, tone: str, saturation: float) -> float:
        """Calculate saturation-based membership"""
        params = self.saturation_params[tone]
        
        if tone == 'warm':
            # Warm undertones benefit from higher saturation
            return min(1.0, saturation * params['sensitivity'] + 0.2)
        elif tone == 'cool':
            # Cool undertones can handle lower saturation
            return max(0.3, 1.0 - abs(saturation - params['threshold']) * params['sensitivity'])
        else:  # neutral
            # Neutral undertones are flexible with saturation
            return 0.6 + 0.4 * (1.0 - abs(saturation - params['threshold']) * params['sensitivity'])
    
    def _normalize_scores(self, scores: Dict[str, float]) -> None:
        """Enhanced normalization with minimum threshold"""
        total_score = sum(scores.values())
        if total_score > 0:
            for tone in scores:
                scores[tone] = max(0.01, scores[tone] / total_score)  # Ensure minimum score
    
    def enhanced_defuzzify(self, fuzzy_scores: Dict[str, float], context: Dict[str, Any]) -> Tuple[str, float, str]:
        """Enhanced defuzzification with confidence calibration"""
        sorted_scores = sorted(fuzzy_scores.items(), key=lambda x: x[1], reverse=True)
        primary_tone = sorted_scores[0][0]
        primary_confidence = sorted_scores[0][1]
        
        # Calculate confidence based on score separation and dominant factors
        if len(sorted_scores) > 1:
            score_separation = sorted_scores[0][1] - sorted_scores[1][1]
            
            # Boost confidence if dominant factors align
            dominant_factors = context.get('dominant_factors', {})
            if dominant_factors.get('strength') == 'high':
                primary_confidence *= 1.2
            elif dominant_factors.get('balance') == 'highly_balanced' and primary_tone == 'neutral':
                primary_confidence *= 1.15
            
            # Adjust for score separation
            if score_separation > 0.2:
                primary_confidence *= 1.1
            elif score_separation < 0.05:
                primary_confidence *= 0.85
        
        # Cap confidence at reasonable maximum
        primary_confidence = min(0.95, primary_confidence)
        
        # Determine confidence level with more nuanced thresholds
        if primary_confidence > 0.70:
            confidence_level = "high"
        elif primary_confidence > 0.55:
            confidence_level = "medium"
        elif primary_confidence > 0.40:
            confidence_level = "low"
        else:
            confidence_level = "mixed"
            # For very low confidence, consider it neutral/mixed
            if primary_confidence < 0.35:
                primary_tone = "neutral"
        
        logger.info(f"Enhanced defuzzification: {primary_tone} ({primary_confidence:.3f}, {confidence_level})")
        
        return primary_tone, primary_confidence, confidence_level
    
    def generate_personalized_explanation(self, undertone: str, context: Dict[str, Any]) -> str:
        """Generate personalized explanation based on dominant factors"""
        dominant_factors = context.get('dominant_factors', {})
        templates = self.explanation_templates[undertone]
        
        # Choose template based on dominant factor
        if dominant_factors.get('primary') == 'rb_dominant':
            template_key = 'rb_dominant'
        elif dominant_factors.get('tertiary') == 'hue_dominant':
            template_key = 'hue_dominant'
        elif dominant_factors.get('secondary') in ['high_saturation', 'low_saturation']:
            template_key = 'saturation_dominant'
        elif dominant_factors.get('balance') in ['highly_balanced', 'moderately_balanced']:
            template_key = 'balanced' if undertone != 'neutral' else 'balanced_ratios'
        else:
            template_key = 'balanced'
        
        # Fallback for neutral undertone
        if undertone == 'neutral' and template_key not in templates:
            template_key = 'versatile'
        
        template = templates.get(template_key, templates.get('balanced', ''))
        
        # Format template with actual values
        try:
            formatted_explanation = template.format(
                rb_diff=context.get('rb_diff', 0),
                rg_ratio=context.get('rg_ratio', 1.0),
                gb_ratio=context.get('gb_ratio', 1.0),
                hue_angle=context.get('hsv_factors', {}).get('hue_angle', 0),
                saturation=context.get('hsv_factors', {}).get('saturation', 0),
                brightness=context.get('brightness', 0.5)
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Template formatting error: {e}")
            formatted_explanation = f"Kulit Anda memiliki undertone {undertone} berdasarkan analisis komprehensif berbagai faktor warna."
        
        return formatted_explanation

class DynamicRecommendationEngine:
    """Enhanced recommendation engine with dynamic palette selection"""
    
    def __init__(self):
        self._init_factor_based_palettes()
        self._init_base_recommendations()
    
    def _init_factor_based_palettes(self):
        """Initialize color palettes based on dominant factors"""
        self.factor_palettes = {
            'cool': {
                'rb_dominant': {
                    'clothing': ["deep navy", "royal blue", "emerald green", "cool gray", "icy lavender"],
                    'accent_colors': ["silver", "white", "pearl"],
                    'makeup': ["berry lipstick", "cool pink blush", "silver eyeshadow"]
                },
                'hue_dominant': {
                    'clothing': ["sapphire blue", "forest green", "plum purple", "charcoal", "mint green"],
                    'accent_colors': ["platinum", "cool silver", "white gold"],
                    'makeup': ["mauve lipstick", "rose blush", "cool-toned foundation"]
                },
                'high_saturation': {
                    'clothing': ["vibrant blue", "jewel green", "rich purple", "bright white", "cool red"],
                    'accent_colors': ["bright silver", "crystal", "diamond"],
                    'makeup': ["bold berry", "bright pink blush", "metallic eyeshadow"]
                },
                'balanced': {
                    'clothing': ["classic navy", "sage green", "soft lavender", "gray", "white"],
                    'accent_colors': ["silver", "platinum", "pearl"],
                    'makeup': ["natural pink", "soft blush", "neutral cool tones"]
                }
            },
            'warm': {
                'rb_dominant': {
                    'clothing': ["rust orange", "warm coral", "terracotta", "golden yellow", "warm brown"],
                    'accent_colors': ["gold", "copper", "bronze"],
                    'makeup': ["coral lipstick", "peach blush", "bronze eyeshadow"]
                },
                'hue_dominant': {
                    'clothing': ["burnt orange", "mustard yellow", "olive green", "warm camel", "paprika"],
                    'accent_colors': ["antique gold", "brass", "warm copper"],
                    'makeup': ["warm orange", "apricot blush", "golden eyeshadow"]
                },
                'high_saturation': {
                    'clothing': ["bright coral", "vibrant orange", "golden yellow", "warm red", "rich brown"],
                    'accent_colors': ["bright gold", "polished copper", "amber"],
                    'makeup': ["bold coral", "bright peach", "golden bronze"]
                },
                'balanced': {
                    'clothing': ["warm beige", "soft peach", "olive", "warm gray", "cream"],
                    'accent_colors': ["gold", "rose gold", "warm brass"],
                    'makeup': ["natural coral", "warm pink", "bronze neutrals"]
                }
            },
            'neutral': {
                'balanced_ratios': {
                    'clothing': ["soft white", "taupe", "blush pink", "mint", "warm gray"],
                    'accent_colors': ["rose gold", "mixed metals", "champagne"],
                    'makeup': ["mlbb (my lips but better)", "natural blush", "neutral eyeshadow"]
                },
                'versatile': {
                    'clothing': ["classic black", "cream", "soft pastels", "earth tones", "jewel tones"],
                    'accent_colors': ["gold and silver", "rose gold", "mixed metals"],
                    'makeup': ["versatile nudes", "adaptable colors", "universal tones"]
                },
                'mixed_signals': {
                    'clothing': ["adaptable colors", "medium tones", "soft brights", "muted colors"],
                    'accent_colors': ["mixed metals", "subtle finishes", "versatile pieces"],
                    'makeup': ["buildable colors", "blendable tones", "universal shades"]
                }
            }
        }
    
    def _init_base_recommendations(self):
        """Initialize base recommendations with enhanced details"""
        self.base_recommendations = {
            "cool": {
                "jewelry": ["silver", "platinum", "white gold", "diamonds"],
                "hair_colors": ["ash blonde", "cool brown", "black", "silver highlights"],
                "nail_colors": ["cool pinks", "berries", "blues", "silver"],
                "fragrance_notes": ["fresh", "aquatic", "green", "citrus"],
                "style_approach": "Classic and refined with cool undertones"
            },
            "warm": {
                "jewelry": ["gold", "copper", "bronze", "amber"],
                "hair_colors": ["golden blonde", "warm brown", "auburn", "honey highlights"],
                "nail_colors": ["corals", "peaches", "warm reds", "gold"],
                "fragrance_notes": ["warm spices", "vanilla", "amber", "woody"],
                "style_approach": "Rich and inviting with golden undertones"
            },
            "neutral": {
                "jewelry": ["rose gold", "mixed metals", "champagne gold", "pearls"],
                "hair_colors": ["balanced tones", "natural shades", "subtle highlights"],
                "nail_colors": ["nudes", "soft pinks", "versatile shades"],
                "fragrance_notes": ["balanced florals", "soft musks", "universal scents"],
                "style_approach": "Versatile and adaptable to various color families"
            }
        }
    
    def get_dynamic_recommendations(self, undertone: str, dominant_factors: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dynamic recommendations based on dominant factors"""
        
        # Determine the best palette based on dominant factors
        palette_key = self._select_palette_key(dominant_factors, undertone)
        
        # Get factor-based palette
        factor_palette = self.factor_palettes[undertone].get(palette_key, 
                                                           self.factor_palettes[undertone]['balanced'])
        
        # Get base recommendations
        base_rec = self.base_recommendations[undertone]
        
        # Combine and enhance recommendations
        recommendations = {
            **factor_palette,
            **base_rec,
            'palette_type': palette_key,
            'confidence_adjustments': self._get_confidence_adjustments(dominant_factors),
            'seasonal_considerations': self._get_seasonal_recommendations(undertone, context),
            'styling_tips': self._get_styling_tips(undertone, dominant_factors)
        }
        
        return recommendations
    
    def _select_palette_key(self, dominant_factors: Dict[str, Any], undertone: str) -> str:
        """Select the most appropriate palette based on dominant factors"""
        if dominant_factors.get('primary') == 'rb_dominant':
            return 'rb_dominant'
        elif dominant_factors.get('tertiary') == 'hue_dominant':
            return 'hue_dominant'
        elif dominant_factors.get('secondary') == 'high_saturation':
            return 'high_saturation'
        elif dominant_factors.get('balance') in ['highly_balanced', 'moderately_balanced']:
            if undertone == 'neutral':
                return 'balanced_ratios'
            else:
                return 'balanced'
        elif undertone == 'neutral':
            return 'versatile'
        else:
            return 'balanced'
    
    def _get_confidence_adjustments(self, dominant_factors: Dict[str, Any]) -> List[str]:
        """Get confidence-based adjustment recommendations"""
        adjustments = []
        
        strength = dominant_factors.get('strength', 'medium')
        if strength == 'high':
            adjustments.append("Anda dapat dengan percaya diri menggunakan warna-warna yang direkomendasikan")
        elif strength == 'low':
            adjustments.append("Mulai dengan warna yang lebih netral, lalu eksplorasi secara bertahap")
        
        if dominant_factors.get('balance') == 'highly_balanced':
            adjustments.append("Fleksibilitas tinggi dalam pemilihan warna memberikan banyak pilihan gaya")
        
        return adjustments
    
    def _get_seasonal_recommendations(self, undertone: str, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get seasonal color recommendations based on undertone and context"""
        brightness = context.get('brightness', 0.5)
        saturation = context.get('hsv_factors', {}).get('saturation', 0.2)
        
        seasonal_recs = {
            'spring': [],
            'summer': [],
            'autumn': [],
            'winter': []
        }
        
        if undertone == 'warm':
            if brightness > 0.6 and saturation > 0.25:  # Light and saturated
                seasonal_recs['spring'] = ["bright corals", "golden yellows", "warm greens", "peach tones"]
                seasonal_recs['autumn'] = ["rich oranges", "deep golds", "warm browns", "spice colors"]
            else:  # Deeper or muted warm
                seasonal_recs['autumn'] = ["burnt oranges", "deep golds", "warm browns", "rich reds"]
                seasonal_recs['spring'] = ["soft peaches", "warm creams", "light corals"]
        
        elif undertone == 'cool':
            if brightness > 0.6:  # Light cool
                seasonal_recs['summer'] = ["soft blues", "lavenders", "cool pinks", "sage greens"]
                seasonal_recs['winter'] = ["icy blues", "cool whites", "deep purples", "true blacks"]
            else:  # Deep cool
                seasonal_recs['winter'] = ["deep blues", "rich purples", "cool reds", "stark whites"]
                seasonal_recs['summer'] = ["muted blues", "soft grays", "dusty roses"]
        
        else:  # neutral
            seasonal_recs['spring'] = ["soft pastels", "warm whites", "light neutrals"]
            seasonal_recs['summer'] = ["cool pastels", "soft grays", "muted tones"]
            seasonal_recs['autumn'] = ["warm earth tones", "muted golds", "soft browns"]
            seasonal_recs['winter'] = ["true colors", "clear tones", "balanced brights"]
        
        return seasonal_recs
    
    def _get_styling_tips(self, undertone: str, dominant_factors: Dict[str, Any]) -> List[str]:
        """Get personalized styling tips based on analysis"""
        tips = []
        
        # Base tips by undertone
        if undertone == 'cool':
            tips.extend([
                "Pilih foundation dengan base pink atau neutral-cool",
                "Gunakan blush dengan undertone pink atau rose",
                "Hindari warna dengan base kuning atau oranye yang kuat"
            ])
        elif undertone == 'warm':
            tips.extend([
                "Pilih foundation dengan base yellow atau golden",
                "Gunakan blush dengan undertone peach atau coral",
                "Hindari warna dengan base pink atau biru yang stark"
            ])
        else:  # neutral
            tips.extend([
                "Anda fleksibel dengan berbagai base foundation",
                "Eksperimen dengan berbagai undertone blush",
                "Fokus pada intensitas warna daripada undertone"
            ])
        
        # Factor-specific tips
        strength = dominant_factors.get('strength', 'medium')
        if strength == 'high':
            tips.append("Undertone Anda sangat jelas, manfaatkan untuk tampilan yang konsisten")
        elif strength == 'low':
            tips.append("Coba berbagai warna untuk menemukan yang paling cocok")
        
        if dominant_factors.get('secondary') == 'high_saturation':
            tips.append("Kulit Anda dapat menahan warna-warna vibrant dan bold")
        elif dominant_factors.get('secondary') == 'low_saturation':
            tips.append("Warna-warna soft dan muted akan terlihat natural pada Anda")
        
        return tips

class FaceDetector:
    """Enhanced face detection with better validation"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Detect faces with multiple detection strategies"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        
        # Try multiple detection parameters
        detection_params = [
            {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)},
            {'scaleFactor': 1.2, 'minNeighbors': 3, 'minSize': (25, 25)},
            {'scaleFactor': 1.3, 'minNeighbors': 4, 'minSize': (20, 20)}
        ]
        
        faces = []
        for params in detection_params:
            faces = self.face_cascade.detectMultiScale(enhanced_gray, **params)
            if len(faces) > 0:
                break
        
        return faces, enhanced_gray
    
    def validate_face(self, face_rect: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> Optional[str]:
        """Enhanced face validation"""
        x, y, w, h = face_rect
        img_height, img_width = image_shape[:2]
        
        # Check aspect ratio
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return "Foto tidak sesuai. Pastikan wajah terlihat dari depan dengan proporsi normal."
        
        # Check minimum size (relative to image)
        min_face_ratio = 0.1  # Face should be at least 10% of image width
        if w < img_width * min_face_ratio or h < img_height * min_face_ratio:
            return "Wajah terlalu kecil dalam foto. Gunakan foto close-up dengan wajah yang lebih besar."
        
        # Check if face is too close to edges
        margin = 0.05  # 5% margin
        if (x < img_width * margin or y < img_height * margin or 
            x + w > img_width * (1 - margin) or y + h > img_height * (1 - margin)):
            return "Wajah terlalu dekat dengan tepi foto. Pastikan wajah berada di tengah frame."
        
        return None
    
    def extract_face_region(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract face region with smart expansion"""
        x, y, w, h = face_rect
        img_height, img_width = image.shape[:2]
        
        # Dynamic expansion based on face size
        expand_factor = max(0.1, min(0.25, 50 / min(w, h)))  # Larger expansion for smaller faces
        
        x_expanded = max(0, int(x - w * expand_factor))
        y_expanded = max(0, int(y - h * expand_factor))
        w_expanded = min(img_width - x_expanded, int(w * (1 + 2 * expand_factor)))
        h_expanded = min(img_height - y_expanded, int(h * (1 + 2 * expand_factor)))
        
        return image[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded]

class EnhancedSkinAnalyzer:
    """Enhanced skin analysis with better region sampling and filtering"""
    
    @staticmethod
    def sample_skin_regions(face_img: np.ndarray) -> List[np.ndarray]:
        """Sample skin regions with adaptive positioning"""
        height, width = face_img.shape[:2]
        
        # Define region templates with relative coordinates
        region_templates = [
            # Forehead regions (3 samples)
            {'name': 'forehead_left', 'coords': (0.1, 0.25, 0.25, 0.45)},
            {'name': 'forehead_center', 'coords': (0.1, 0.25, 0.4, 0.6)},
            {'name': 'forehead_right', 'coords': (0.1, 0.25, 0.55, 0.75)},
            
            # Cheek regions (2 samples)
            {'name': 'cheek_left', 'coords': (0.35, 0.55, 0.1, 0.35)},
            {'name': 'cheek_right', 'coords': (0.35, 0.55, 0.65, 0.9)},
            
            # Nose bridge (1 sample)
            {'name': 'nose_bridge', 'coords': (0.3, 0.5, 0.4, 0.6)},
            
            # Chin area (1 sample)
            {'name': 'chin', 'coords': (0.7, 0.85, 0.35, 0.65)},
            
            # Additional temple regions for better sampling
            {'name': 'temple_left', 'coords': (0.2, 0.4, 0.05, 0.25)},
            {'name': 'temple_right', 'coords': (0.2, 0.4, 0.75, 0.95)}
        ]
        
        regions = []
        for template in region_templates:
            y1_rel, y2_rel, x1_rel, x2_rel = template['coords']
            
            y1 = int(height * y1_rel)
            y2 = int(height * y2_rel)
            x1 = int(width * x1_rel)
            x2 = int(width * x2_rel)
            
            # Ensure coordinates are within bounds
            y1, y2 = max(0, y1), min(height, y2)
            x1, x2 = max(0, x1), min(width, x2)
            
            if y2 > y1 and x2 > x1:  # Valid region
                region = face_img[y1:y2, x1:x2]
                if region.size > 0:
                    regions.append(region)
        
        return regions
    
    @staticmethod
    def create_enhanced_skin_mask(hsv_region: np.ndarray) -> np.ndarray:
        """Create enhanced skin mask with multiple color spaces"""
        # Convert to different color spaces for better skin detection
        ycrcb_region = cv2.cvtColor(cv2.cvtColor(hsv_region, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2YCrCb)
        
        # HSV-based skin detection
        lower_skin_hsv1 = np.array([0, 10, 60], dtype=np.uint8)
        upper_skin_hsv1 = np.array([25, 255, 255], dtype=np.uint8)
        mask_hsv1 = cv2.inRange(hsv_region, lower_skin_hsv1, upper_skin_hsv1)
        
        lower_skin_hsv2 = np.array([0, 15, 30], dtype=np.uint8)
        upper_skin_hsv2 = np.array([35, 200, 200], dtype=np.uint8)
        mask_hsv2 = cv2.inRange(hsv_region, lower_skin_hsv2, upper_skin_hsv2)
        
        # YCrCb-based skin detection
        lower_skin_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb_region, lower_skin_ycrcb, upper_skin_ycrcb)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        combined_mask = cv2.bitwise_or(combined_mask, mask_ycrcb)
        
        # Morphological operations for cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        return combined_mask
    
    def analyze_skin_color(self, skin_regions: List[np.ndarray]) -> Optional[Tuple[int, int, int]]:
        """Enhanced skin color analysis with statistical filtering"""
        all_pixels = []
        region_averages = []
        
        for i, region in enumerate(skin_regions):
            if region.size == 0:
                continue
                
            hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            enhanced_mask = self.create_enhanced_skin_mask(hsv_region)
            skin_pixels = region[enhanced_mask > 0]
            
            if len(skin_pixels) > 20:  # Minimum pixel threshold
                region_avg = np.mean(skin_pixels, axis=0)
                region_averages.append(region_avg)
                all_pixels.extend(skin_pixels)
        
        if len(all_pixels) < 100:  # Increased minimum threshold
            return None
        
        # Use both individual pixels and region averages for robustness
        final_color = self._compute_robust_average(np.array(all_pixels), region_averages)
        return final_color
    
    def _compute_robust_average(self, all_pixels: np.ndarray, region_averages: List[np.ndarray]) -> Tuple[int, int, int]:
        """Compute robust average using multiple statistical methods"""
        # Method 1: Filtered pixel-wise average
        pixel_avg = self._filter_and_average_pixels(all_pixels)
        
        # Method 2: Region-wise average
        if region_averages:
            region_avg = tuple(map(int, np.mean(region_averages, axis=0)))
        else:
            region_avg = pixel_avg
        
        # Method 3: Median-based approach
        median_color = tuple(map(int, np.median(all_pixels, axis=0)))
        
        # Weight the different methods
        weights = [0.5, 0.3, 0.2]  # Pixel-wise, region-wise, median
        colors = [pixel_avg, region_avg, median_color]
        
        final_color = tuple(
            int(sum(w * c[i] for w, c in zip(weights, colors)))
            for i in range(3)
        )
        
        return final_color
    
    def _filter_and_average_pixels(self, pixels: np.ndarray) -> Tuple[int, int, int]:
        """Enhanced pixel filtering with multiple outlier removal methods"""
        if len(pixels) < 10:
            return tuple(map(int, np.mean(pixels, axis=0)))
        
        # Method 1: IQR-based filtering
        q25 = np.percentile(pixels, 25, axis=0)
        q75 = np.percentile(pixels, 75, axis=0)
        iqr = q75 - q25
        
        # Adaptive IQR multiplier based on data distribution
        iqr_multiplier = 1.5 if len(pixels) > 1000 else 2.0
        
        mask_iqr = np.all(
            (pixels >= q25 - iqr_multiplier * iqr) & 
            (pixels <= q75 + iqr_multiplier * iqr), axis=1
        )
        
        # Method 2: Z-score based filtering
        mean_color = np.mean(pixels, axis=0)
        std_color = np.std(pixels, axis=0)
        z_scores = np.abs((pixels - mean_color) / (std_color + 1e-8))
        mask_zscore = np.all(z_scores < 2.5, axis=1)
        
        # Combine filters
        combined_mask = mask_iqr & mask_zscore
        filtered_pixels = pixels[combined_mask]
        
        if len(filtered_pixels) > 10:
            return tuple(map(int, np.mean(filtered_pixels, axis=0)))
        else:
            # Fallback to less strict filtering
            return tuple(map(int, np.mean(pixels[mask_iqr], axis=0)))

class EnhancedSkinToneDetector:
    """Main enhanced skin tone detection service"""
    
    def __init__(self):
        self.face_detector = FaceDetector()
        self.skin_analyzer = EnhancedSkinAnalyzer()
        self.fuzzy_analyzer = EnhancedFuzzyUndertoneAnalyzer()
        self.recommendation_engine = DynamicRecommendationEngine()
    
    def detect_skin_tone(self, image_path: str) -> Dict[str, Any]:
        """Enhanced main skin tone detection method"""
        try:
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                return {"error": "Failed to load image"}
            
            faces, enhanced_gray = self.face_detector.detect_faces(image)
            if len(faces) == 0:
                return {"error": "No face detected. Pastikan wajah terlihat jelas dalam foto dengan pencahayaan yang baik."}
            
            # Use largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Enhanced validation
            validation_error = self.face_detector.validate_face(largest_face, image.shape)
            if validation_error:
                return {"error": validation_error}
            
            # Extract face region
            face_region = self.face_detector.extract_face_region(image, largest_face)
            
            # Save cropped face
            cropped_path = self._save_cropped_face(face_region, image_path)
            
            # Enhanced skin color analysis
            skin_regions = self.skin_analyzer.sample_skin_regions(face_region)
            color_avg = self.skin_analyzer.analyze_skin_color(skin_regions)
            
            if not color_avg:
                return {"error": "Tidak dapat menganalisa warna kulit. Coba foto dengan pencahayaan yang lebih baik dan pastikan wajah terlihat jelas."}
            
            # Enhanced fuzzy analysis
            b_avg, g_avg, r_avg = color_avg
            fuzzy_scores, context = self.fuzzy_analyzer.enhanced_fuzzy_inference(r_avg, g_avg, b_avg)
            undertone, confidence, confidence_level = self.fuzzy_analyzer.enhanced_defuzzify(fuzzy_scores, context)
            
            # Generate personalized explanation
            personalized_reason = self.fuzzy_analyzer.generate_personalized_explanation(undertone, context)
            
            # Get dynamic recommendations
            dynamic_recommendations = self.recommendation_engine.get_dynamic_recommendations(
                undertone, context.get('dominant_factors', {}), context
            )
            
            return self._build_enhanced_analysis_result(
                r_avg, g_avg, b_avg, undertone, confidence, confidence_level,
                fuzzy_scores, context, cropped_path, personalized_reason, dynamic_recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced skin tone detection: {str(e)}")
            return {"error": f"Terjadi kesalahan dalam analisis: {str(e)}"}
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Enhanced image loading with better preprocessing"""
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        # Resize if too large
        height, width = image.shape[:2]
        if max(height, width) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)), 
                             interpolation=cv2.INTER_LANCZOS4)
        
        # Apply subtle enhancement for better analysis
        # Convert to LAB for better color processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_image
    
    def _save_cropped_face(self, face_region: np.ndarray, original_path: str) -> str:
        """Save cropped face with unique naming"""
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        timestamp = int(time.time())
        cropped_filename = f"{base_name}_cropped_{timestamp}.jpg"
        cropped_path = os.path.join(Config.CROPPED_FOLDER, cropped_filename)
        
        # Save with high quality
        cv2.imwrite(cropped_path, face_region, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return f"uploads/cropped/{cropped_filename}"
    
    def _build_enhanced_analysis_result(
        self, r_avg: int, g_avg: int, b_avg: int, undertone: str,
        confidence: float, confidence_level: str, fuzzy_scores: Dict[str, float],
        context: Dict[str, Any], cropped_path: str, personalized_reason: str,
        dynamic_recommendations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build enhanced analysis result dictionary"""
        return {
            "overtone": (r_avg, g_avg, b_avg),
            "rgb_color": f"rgb({r_avg}, {g_avg}, {b_avg})",
            "hex_color": f"#{r_avg:02x}{g_avg:02x}{b_avg:02x}",
            "undertone": undertone,
            "skin_tone": undertone,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "fuzzy_scores": fuzzy_scores,
            "dominant_factors": context.get('dominant_factors', {}),
            "context": context,
            "cropped_image": cropped_path,
            "personalized_reason": personalized_reason,
            "dynamic_recommendations": dynamic_recommendations,
            "statistics": {
                "brightness": context['brightness'],
                "saturation": context.get('hsv_factors', {}).get('saturation', 0),
                "hue_angle": context.get('hsv_factors', {}).get('hue_angle', 0),
                "rb_difference": context['rb_diff'],
                "rg_ratio": context['rg_ratio'],
                "gb_ratio": context['gb_ratio'],
                "chromatic_intensity": context.get('advanced_factors', {}).get('chromatic_intensity', 0)
            }
        }

# Flask Application Setup with Enhanced Features
def create_app() -> Flask:
    """Enhanced application factory"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CROPPED_FOLDER'], exist_ok=True)
    
    # Initialize enhanced services
    skin_detector = EnhancedSkinToneDetector()
    
    def allowed_file(filename: str) -> bool:
        """Check if file extension is allowed"""
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)
    
    @app.route('/', methods=['GET', 'POST'])
    def index():
        """Enhanced main route handler"""
        if request.method == 'GET':
            return render_template('index.html')
        
        # Handle POST request
        if 'file' not in request.files:
            flash("Tidak ada file yang dipilih", "error")
            return render_template('index.html', error="Tidak ada file yang dipilih")
        
        file = request.files['file']
        
        if file.filename == '':
            flash("Tidak ada file yang dipilih", "error")
            return render_template('index.html', error="Tidak ada file yang dipilih")
        
        if not allowed_file(file.filename):
            error_msg = "Format file tidak diizinkan. Gunakan PNG, JPG, atau JPEG"
            flash(error_msg, "error")
            return render_template('index.html', error=error_msg)
        
        try:
            # Save uploaded file with unique name
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            unique_id = uuid.uuid4().hex[:8]
            unique_filename = f"{timestamp}_{unique_id}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            logger.info(f"File saved: {filepath}")
            
            # Enhanced analysis
            analysis = skin_detector.detect_skin_tone(filepath)
            
            if "error" in analysis:
                flash(analysis["error"], "error")
                # Clean up uploaded file on error
                try:
                    os.remove(filepath)
                except:
                    pass
                return render_template('index.html', error=analysis["error"])
            
            # Enhanced success message
            confidence_text = {
                'high': 'tinggi', 'medium': 'sedang', 'low': 'rendah', 'mixed': 'campuran'
            }.get(analysis['confidence_level'], 'sedang')
            
            success_msg = (
                f"✨ Analisis berhasil! Undertone: {analysis['undertone'].title()} "
                f"(Kepercayaan {confidence_text}: {analysis['confidence']:.1%})"
            )
            flash(success_msg, "success")
            
            # Log successful analysis with enhanced details
            dominant_factors = analysis.get('dominant_factors', {})
            logger.info(f"Enhanced analysis completed: {analysis['undertone']} "
                       f"(confidence: {analysis['confidence']:.3f}, "
                       f"dominant factors: {dominant_factors})")
            
            return render_template('index.html', 
                                 result=analysis.get('dynamic_recommendations', {}), 
                                 analysis=analysis)
            
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            error_msg = f"Terjadi kesalahan: {str(e)}"
            flash(error_msg, "error")
            # Clean up uploaded file on error
            try:
                if 'filepath' in locals():
                    os.remove(filepath)
            except:
                pass
            return render_template('index.html', error=error_msg)
    
    @app.route('/api/analyze', methods=['POST'])
    def api_analyze():
        """Enhanced API endpoint for skin tone analysis"""
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format"}), 400
        
        try:
            # Save file
            filename = secure_filename(file.filename)
            unique_filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Enhanced analysis
            analysis = skin_detector.detect_skin_tone(filepath)
            
            if "error" in analysis:
                return jsonify({"error": analysis["error"]}), 400
            
            return jsonify({
                "success": True,
                "analysis": analysis,
                "recommendations": analysis.get('dynamic_recommendations', {}),
                "explanation": analysis.get('personalized_reason', ''),
                "confidence": {
                    "level": analysis['confidence_level'],
                    "score": analysis['confidence'],
                    "factors": analysis.get('dominant_factors', {})
                }
            })
        
        except Exception as e:
            logger.error(f"Enhanced API error: {str(e)}")
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    
    @app.errorhandler(413)
    def too_large(e):
        """Handle file too large error"""
        flash("File terlalu besar. Maksimal 16MB", "error")
        return redirect(url_for('index')), 413
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle server errors"""
        logger.error(f"Server error: {str(e)}")
        flash("Terjadi kesalahan server. Silakan coba lagi nanti.", "error")
        return render_template('index.html', error="Server error"), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)