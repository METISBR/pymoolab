# Author: Professor Thiago Santos at UFOP, Brazil
# -*- coding: utf-8 -*-
"""
Centralized styles and colors system for PymooLab application.
All colors, fonts and styles are defined here and aligned with the PymooLab logo palette.

UTF-8 without BOM - Central style system for PymooLab
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class ColorPalette:
    """
    Application color palette.
    Aligned with PymooLab logo palette (cyan-blue + steel gray).
    
    Main colors:
    - primary: Logo blue (~#10A0E0)
    - background: Light gray (#FAFAFA)
    - surface: White (#FFFFFF)
    - text: Dark gray (#212121)
    """

    # Primary colors - PymooLab logo blue
    primary: str = "#10A0E0"
    primary_dark: str = "#0B7DB8"
    primary_light: str = "#39B7F0"
    
    # State colors
    success: str = "#43A047"
    warning: str = "#FB8C00"
    # Replace red danger with neutral gray per UI direction
    danger: str = "#6B7280"
    info: str = "#1E88E5"
    
    # Neutral colors (light theme)
    background: str = "#FAFAFA"
    surface: str = "#FFFFFF"
    surface_variant: str = "#F5F5F5"
    
    # Text colors
    text_primary: str = "#212121"
    text_secondary: str = "#757575"
    text_disabled: str = "#9E9E9E"
    text_on_primary: str = "#FFFFFF"
    
    # Borders and dividers
    border: str = "#E0E0E0"
    border_light: str = "#EEEEEE"
    divider: str = "#E0E0E0"
    
    # List and highlight colors (logo-driven)
    accent_blue: str = "#10A0E0"
    accent_orange: str = "#5F6E7A"   # gray replacement for prior warm/red emphasis
    selection_blue: str = "#0B7DB8"
    selection_orange: str = "#475569"  # gray selection (slate)
    
    def to_dict(self) -> Dict[str, str]:
        """Return palette as dictionary for use with qt_material."""
        return {
            'primary': self.primary,
            'primary_dark': self.primary_dark,
            'primary_light': self.primary_light,
            'success': self.success,
            'warning': self.warning,
            'danger': self.danger,
            'info': self.info,
            'background': self.background,
            'surface': self.surface,
            'text_primary': self.text_primary,
            'text_secondary': self.text_secondary,
        }


@dataclass
class Typography:
    """Typography configuration."""
    
    # Font families
    font_family: str = "Segoe UI"
    font_family_mono: str = "Consolas"
    
    # Font sizes
    font_size_xs: int = 10
    font_size_sm: int = 11
    font_size_base: int = 12
    font_size_lg: int = 14
    font_size_xl: int = 16
    font_size_2xl: int = 20
    font_size_3xl: int = 24
    
    # Font weights
    font_weight_normal: int = 400
    font_weight_medium: int = 500
    font_weight_semibold: int = 600
    font_weight_bold: int = 700


@dataclass
class Spacing:
    """Spacing system."""
    
    xs: int = 4
    sm: int = 6
    md: int = 8
    lg: int = 12
    xl: int = 16
    xxl: int = 24
    xxxl: int = 32


@dataclass
class AnimationSettings:
    """Animation configuration."""
    
    duration_fast: int = 150
    duration_normal: int = 300
    duration_slow: int = 500
    
    # Easing curves (QEasingCurve.Type)
    easing_default: str = "InOutCubic"
    easing_entrance: str = "OutCubic"
    easing_exit: str = "InCubic"


class AppStyles:
    """
    Main styles class for PymooLab application.
    Use as single source of truth for all styles.
    
    This class centralizes all colors, fonts and styles
    to ensure visual consistency across the application.
    
    Usage:
        from styles import AppStyles
        
        # Colors
        color = AppStyles.colors.primary
        
        # Stylesheet
        widget.setStyleSheet(f"background: {AppStyles.colors.surface};")
    """
    
    colors = ColorPalette()
    typography = Typography()
    spacing = Spacing()
    animation = AnimationSettings()
    
    @staticmethod
    def get_stylesheet() -> str:
        """
        Return complete CSS stylesheet for the application.
        Complementary to the application theme.
        """
        colors = AppStyles.colors
        typo = AppStyles.typography
        
        return f"""
        /* Complementary styles for PymooLab */
        
        QMainWindow, QWidget {{
            background-color: {colors.background};
            color: {colors.text_primary};
            font-family: '{typo.font_family}';
            font-size: {typo.font_size_base}px;
        }}
        
        QGroupBox {{
            font-weight: {typo.font_weight_semibold};
            color: {colors.text_primary};
            border: 1px solid {colors.border};
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 8px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            color: {colors.primary};
        }}
        
        QListWidget::item {{
            padding: 4px 8px;
            border-radius: 4px;
        }}
        
        QListWidget::item:selected {{
            background-color: {colors.primary};
            color: {colors.text_on_primary};
        }}
        
        QListWidget::item:hover {{
            background-color: {colors.surface_variant};
        }}
        
        QComboBox {{
            padding: 6px 12px;
            border: 1px solid {colors.border};
            border-radius: 4px;
            background-color: {colors.surface};
        }}
        
        QComboBox:focus {{
            border: 2px solid {colors.primary};
        }}
        
        QSpinBox {{
            padding: 6px 8px;
            border: 1px solid {colors.border};
            border-radius: 4px;
            background-color: {colors.surface};
        }}
        
        QSpinBox:focus {{
            border: 2px solid {colors.primary};
        }}
        
        QLineEdit {{
            padding: 8px 12px;
            border: 1px solid {colors.border};
            border-radius: 4px;
            background-color: {colors.surface};
        }}
        
        QLineEdit:focus {{
            border: 2px solid {colors.primary};
        }}
        
        QPlainTextEdit {{
            border: 1px solid {colors.border};
            border-radius: 4px;
            background-color: {colors.surface};
        }}
        
        QProgressBar {{
            border: none;
            border-radius: 4px;
            text-align: center;
            background-color: {colors.surface_variant};
        }}
        
        QProgressBar::chunk {{
            background-color: {colors.primary};
            border-radius: 4px;
        }}
        
        QSplitter::handle {{
            background-color: {colors.border};
        }}
        
        QTabWidget::pane {{
            border: 1px solid {colors.border};
            border-radius: 4px;
            background-color: {colors.surface};
        }}
        
        QTabBar::tab {{
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {colors.surface};
            border-bottom: 2px solid {colors.primary};
        }}
        
        QTabBar::tab:!selected {{
            background-color: {colors.surface_variant};
        }}
        """
    
    @staticmethod
    def get_qt_material_theme() -> Dict[str, str]:
        """
        Return extra configuration for qt_material.
        Use with apply_stylesheet(app, extra=AppStyles.get_qt_material_theme())
        """
        return AppStyles.colors.to_dict()
    
    @staticmethod
    def get_algorithm_list_style() -> str:
        """Return stylesheet for algorithm list."""
        colors = AppStyles.colors
        return f"""
            QListWidget::item {{
                color: {colors.accent_blue};
                padding: 2px 4px;
            }}
            QListWidget::item:selected {{
                background: {colors.selection_blue};
                color: #f8fafc;
            }}
        """
    
    @staticmethod
    def get_problem_list_style() -> str:
        """Return stylesheet for problem list."""
        colors = AppStyles.colors
        return f"""
            QListWidget::item {{
                color: {colors.accent_orange};
                padding: 2px 4px;
            }}
            QListWidget::item:selected {{
                background: {colors.selection_orange};
                color: #fff7ed;
            }}
        """
    
    @staticmethod
    def get_selection_card_style(color: str = None) -> str:
        """
        Return stylesheet for selection cards.
        
        Args:
            color: Background color (default: primary)
        """
        colors = AppStyles.colors
        bg_color = color or colors.primary
        return f"""
            background: {bg_color};
            color: {colors.text_on_primary};
            border-radius: 6px;
            padding: 6px;
            font-weight: 700;
        """
    
    @staticmethod
    def get_title_style() -> str:
        """Return stylesheet for titles."""
        colors = AppStyles.colors
        return f"color: {colors.accent_blue}; font-weight: 700;"
    
    @staticmethod
    def get_muted_style() -> str:
        """Return stylesheet for secondary/muted text."""
        colors = AppStyles.colors
        return f"color: {colors.text_secondary};"


# Export for easy import
__all__ = ['AppStyles', 'ColorPalette', 'Typography', 'Spacing', 'AnimationSettings']
