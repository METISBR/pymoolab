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
    - primary: Logo blue/cyan
    - background: Cool application canvas
    - surface: White panels
    - text: Slate foreground
    """

    # Primary colors - PymooLab logo blue
    primary: str = "#0A84FF"
    primary_dark: str = "#0369A1"
    primary_light: str = "#38BDF8"
    
    # State colors
    success: str = "#16A34A"
    warning: str = "#D97706"
    danger: str = "#DC2626"
    info: str = "#0284C7"
    
    # Neutral colors (light theme)
    background: str = "#F6F8FB"
    surface: str = "#FFFFFF"
    surface_variant: str = "#F1F5F9"
    surface_soft: str = "#F8FAFC"
    
    # Text colors
    text_primary: str = "#0F172A"
    text_secondary: str = "#475569"
    text_disabled: str = "#94A3B8"
    text_on_primary: str = "#FFFFFF"
    
    # Borders and dividers
    border: str = "#D8E1EA"
    border_light: str = "#E6EDF5"
    divider: str = "#E2E8F0"
    
    # List and highlight colors (logo-driven)
    accent_blue: str = "#0A84FF"
    accent_orange: str = "#475569"
    selection_blue: str = "#0369A1"
    selection_orange: str = "#334155"

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
        /* PymooLab modern scientific workstation shell */

        QMainWindow, QWidget {{
            background-color: {colors.background};
            color: {colors.text_primary};
            font-family: '{typo.font_family}';
            font-size: {typo.font_size_base}px;
        }}

        QMenuBar {{
            background-color: {colors.surface};
            color: {colors.text_secondary};
            border-bottom: 1px solid {colors.border_light};
        }}

        QMenuBar::item:selected {{
            background-color: {colors.surface_variant};
            color: {colors.text_primary};
        }}

        QScrollArea {{
            border: 0;
            background: transparent;
        }}

        QGroupBox {{
            background-color: {colors.surface};
            color: {colors.text_primary};
            border: 1px solid {colors.border_light};
            border-radius: 8px;
            margin-top: 14px;
            padding: 14px 10px 10px 10px;
            font-weight: {typo.font_weight_semibold};
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 6px;
            color: {colors.text_primary};
            background-color: {colors.surface};
        }}

        QLabel {{
            color: {colors.text_primary};
            background: transparent;
        }}

        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            min-height: 28px;
            padding: 5px 10px;
            border: 1px solid {colors.border};
            border-radius: 6px;
            background-color: {colors.surface};
            color: {colors.text_primary};
            selection-background-color: {colors.primary};
            selection-color: {colors.text_on_primary};
        }}

        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {colors.primary};
            background-color: #FFFFFF;
        }}

        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
            color: {colors.text_disabled};
            background-color: {colors.surface_variant};
        }}

        QPlainTextEdit, QTextEdit {{
            border: 1px solid {colors.border};
            border-radius: 8px;
            background-color: {colors.surface};
            color: {colors.text_primary};
            padding: 6px;
            selection-background-color: {colors.primary};
            selection-color: {colors.text_on_primary};
        }}

        QListWidget {{
            border: 1px solid {colors.border_light};
            border-radius: 8px;
            background-color: {colors.surface};
            alternate-background-color: {colors.surface_soft};
            outline: 0;
        }}

        QListWidget::item {{
            min-height: 24px;
            padding: 4px 8px;
            border-radius: 5px;
            margin: 1px 3px;
        }}

        QListWidget::item:hover {{
            background-color: {colors.surface_variant};
        }}

        QListWidget::item:selected {{
            background-color: #E5F2FF;
            color: {colors.primary_dark};
        }}

        QPushButton {{
            min-height: 30px;
            padding: 6px 12px;
            border: 1px solid {colors.border};
            border-radius: 7px;
            background-color: {colors.surface};
            color: {colors.text_primary};
            font-weight: {typo.font_weight_medium};
        }}

        QPushButton:hover {{
            background-color: {colors.surface_variant};
            border-color: {colors.primary_light};
        }}

        QPushButton:pressed {{
            background-color: #E5F2FF;
        }}

        QPushButton:disabled {{
            background-color: {colors.surface_variant};
            color: {colors.text_disabled};
            border-color: {colors.border_light};
        }}

        QProgressBar {{
            min-height: 9px;
            border: 0;
            border-radius: 5px;
            text-align: center;
            background-color: #E2E8F0;
            color: {colors.text_secondary};
        }}

        QProgressBar::chunk {{
            background-color: {colors.primary};
            border-radius: 5px;
        }}

        QSplitter::handle {{
            background-color: transparent;
        }}

        QSplitter::handle:hover {{
            background-color: {colors.border_light};
        }}

        QTableWidget {{
            border: 1px solid {colors.border_light};
            border-radius: 8px;
            background-color: {colors.surface};
            gridline-color: {colors.border_light};
            alternate-background-color: {colors.surface_soft};
            selection-background-color: #E5F2FF;
            selection-color: {colors.text_primary};
        }}

        QTableWidget::item {{
            padding: 4px 6px;
            border: 0;
        }}

        QHeaderView::section {{
            background-color: {colors.surface_soft};
            color: {colors.text_secondary};
            border: 0;
            border-bottom: 1px solid {colors.border_light};
            padding: 6px;
            font-weight: {typo.font_weight_semibold};
        }}

        QTabWidget#primaryWorkflowTabs {{
            background-color: {colors.background};
        }}

        QTabWidget#primaryWorkflowTabs::pane {{
            border: 0;
            border-top: 1px solid {colors.border_light};
            background-color: {colors.background};
            margin: 0;
        }}

        QTabWidget#primaryWorkflowTabs::tab-bar {{
            alignment: center;
            background-color: {colors.surface};
        }}

        QTabBar#primaryWorkflowTabBar {{
            background-color: {colors.surface};
            min-height: 64px;
            border: 0;
            border-bottom: 1px solid {colors.border_light};
        }}

        QTabBar#primaryWorkflowTabBar::tab {{
            background: transparent;
            border: 0;
            margin: 0;
            padding: 0;
            min-height: 64px;
        }}

        QToolTip {{
            background-color: {colors.text_primary};
            color: {colors.text_on_primary};
            border: 0;
            border-radius: 6px;
            padding: 6px 8px;
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
    def get_app_shell_stylesheet() -> str:
        """Return the compact top-shell stylesheet contract."""
        colors = AppStyles.colors
        return (
            f"background-color: {colors.surface}; "
            f"border-bottom: 1px solid {colors.border_light};"
        )
    
    @staticmethod
    def get_algorithm_list_style() -> str:
        """Return stylesheet for algorithm list."""
        colors = AppStyles.colors
        return f"""
            QListWidget {{
                border: 1px solid {colors.border_light};
                border-radius: 8px;
                background: {colors.surface};
            }}
            QListWidget::item {{
                color: {colors.text_primary};
                padding: 4px 8px;
                margin: 1px 3px;
                border-radius: 5px;
            }}
            QListWidget::item:hover {{
                background: {colors.surface_variant};
            }}
            QListWidget::item:selected {{
                background: #E5F2FF;
                color: {colors.selection_blue};
                font-weight: 600;
            }}
        """
    
    @staticmethod
    def get_problem_list_style() -> str:
        """Return stylesheet for problem list."""
        colors = AppStyles.colors
        return f"""
            QListWidget {{
                border: 1px solid {colors.border_light};
                border-radius: 8px;
                background: {colors.surface};
            }}
            QListWidget::item {{
                color: {colors.text_primary};
                padding: 4px 8px;
                margin: 1px 3px;
                border-radius: 5px;
            }}
            QListWidget::item:hover {{
                background: {colors.surface_variant};
            }}
            QListWidget::item:selected {{
                background: {colors.surface_variant};
                color: {colors.selection_orange};
                font-weight: 600;
            }}
        """

    @staticmethod
    def get_metric_list_style() -> str:
        """Return stylesheet for metric list."""
        colors = AppStyles.colors
        return f"""
            QListWidget {{
                border: 1px solid {colors.border_light};
                border-radius: 8px;
                background: {colors.surface};
            }}
            QListWidget::item {{
                color: {colors.text_primary};
                padding: 4px 8px;
                margin: 1px 3px;
                border-radius: 5px;
            }}
            QListWidget::item:hover {{
                background: {colors.surface_variant};
            }}
            QListWidget::item:selected {{
                background: #DCFCE7;
                color: {colors.success};
                font-weight: 600;
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
            border-radius: 8px;
            padding: 8px;
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

    @staticmethod
    def get_helper_text_style() -> str:
        """Return stylesheet for short helper text below dense controls."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return f"color: {colors.text_secondary}; font-size: {typo.font_size_sm}px;"

    @staticmethod
    def get_section_subtitle_style() -> str:
        """Return stylesheet for section subtitles and explanatory copy."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return f"color: {colors.text_secondary}; font-size: {typo.font_size_base}px;"

    @staticmethod
    def get_count_badge_style() -> str:
        """Return stylesheet for compact counters beside list headers."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return (
            f"color: {colors.text_secondary}; "
            f"font-size: {typo.font_size_sm}px; "
            f"padding: 2px 7px; "
            f"border: 1px solid {colors.border_light}; "
            f"border-radius: 999px; "
            f"background: {colors.surface_soft};"
        )

    @staticmethod
    def get_panel_hint_style() -> str:
        """Return stylesheet for wrapped hints inside panels."""
        colors = AppStyles.colors
        return (
            f"color: {colors.text_secondary}; "
            f"background: {colors.surface_soft}; "
            f"border: 1px solid {colors.border_light}; "
            f"border-radius: 8px; "
            f"padding: 8px;"
        )

    @staticmethod
    def get_guided_panel_style() -> str:
        """Return stylesheet for beginner-facing review panels."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return (
            f"color: {colors.text_primary}; "
            f"font-size: {typo.font_size_base}px; "
            f"background: {colors.surface}; "
            f"border: 1px solid {colors.border_light}; "
            f"border-left: 4px solid {colors.primary}; "
            f"border-radius: 8px; "
            f"padding: 8px;"
        )

    @staticmethod
    def get_primary_button_style() -> str:
        """Return stylesheet for primary workflow actions."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return f"""
            QPushButton {{
                background-color: {colors.primary};
                color: {colors.text_on_primary};
                border: 1px solid {colors.primary_dark};
                border-radius: 7px;
                padding: 8px 14px;
                font-weight: {typo.font_weight_semibold};
            }}
            QPushButton:hover {{
                background-color: {colors.primary_dark};
            }}
            QPushButton:disabled {{
                background-color: {colors.border};
                color: {colors.text_disabled};
                border-color: {colors.border_light};
            }}
        """

    @staticmethod
    def get_secondary_button_style() -> str:
        """Return stylesheet for secondary actions."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return f"""
            QPushButton {{
                background-color: {colors.surface};
                color: {colors.text_primary};
                border: 1px solid {colors.border};
                border-radius: 7px;
                padding: 7px 12px;
                font-weight: {typo.font_weight_medium};
            }}
            QPushButton:hover {{
                background-color: {colors.surface_variant};
                border-color: {colors.primary_light};
            }}
            QPushButton:disabled {{
                background-color: {colors.surface_variant};
                color: {colors.text_disabled};
                border-color: {colors.border_light};
            }}
        """

    @staticmethod
    def get_danger_button_style() -> str:
        """Return stylesheet for destructive actions."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return f"""
            QPushButton {{
                background-color: #FEF2F2;
                color: {colors.danger};
                border: 1px solid #FECACA;
                border-radius: 7px;
                padding: 7px 12px;
                font-weight: {typo.font_weight_semibold};
            }}
            QPushButton:hover {{
                background-color: #FEE2E2;
                border-color: #FCA5A5;
            }}
            QPushButton:disabled {{
                background-color: {colors.surface_variant};
                color: {colors.text_disabled};
                border-color: {colors.border_light};
            }}
        """

    @staticmethod
    def get_status_pill_style(color: str = None) -> str:
        """Return compact status-pill styling."""
        colors = AppStyles.colors
        fg = color or colors.success
        return (
            f"color: {fg}; "
            f"background: {colors.surface_soft}; "
            f"border: 1px solid {colors.border_light}; "
            f"border-radius: 999px; "
            f"padding: 3px 8px; "
            "font-weight: 600;"
        )

    @staticmethod
    def get_metric_tile_style(accent: str = None) -> str:
        """Return summary metric tile styling."""
        colors = AppStyles.colors
        left = accent or colors.primary
        return (
            f"color: {colors.text_primary}; "
            f"background: {colors.surface}; "
            f"border: 1px solid {colors.border_light}; "
            f"border-left: 4px solid {left}; "
            f"border-radius: 8px; "
            f"padding: 10px;"
        )

    @staticmethod
    def get_log_view_style() -> str:
        """Return stylesheet for technical log panes."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return (
            f"font-family: '{typo.font_family_mono}'; "
            f"font-size: {typo.font_size_sm}px; "
            f"color: {colors.text_secondary}; "
            f"background: {colors.surface_soft}; "
            f"border: 1px solid {colors.border_light}; "
            "border-radius: 8px;"
        )

    @staticmethod
    def get_table_style(object_name: str = None) -> str:
        """Return table stylesheet, optionally scoped to an object name."""
        colors = AppStyles.colors
        selector = f"QTableWidget#{object_name}" if object_name else "QTableWidget"
        return f"""
            {selector} {{
                border: 1px solid {colors.border_light};
                border-radius: 8px;
                background-color: {colors.surface};
                gridline-color: {colors.border_light};
                alternate-background-color: {colors.surface_soft};
                selection-background-color: #E5F2FF;
                selection-color: {colors.text_primary};
            }}
            {selector}::item {{
                border: none;
                padding: 4px 6px;
            }}
            {selector}::item:selected,
            {selector}::item:selected:active,
            {selector}::item:selected:!active,
            {selector}::item:hover {{
                background-color: #E5F2FF;
                color: {colors.text_primary};
            }}
            {selector} QHeaderView::section {{
                background-color: {colors.surface_soft};
                color: {colors.text_secondary};
                border: 0;
                border-bottom: 1px solid {colors.border_light};
                padding: 6px;
                font-weight: 600;
            }}
        """


# Export for easy import
__all__ = ['AppStyles', 'ColorPalette', 'Typography', 'Spacing', 'AnimationSettings']
