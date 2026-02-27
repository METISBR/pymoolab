# -*- coding: utf-8 -*-
"""
Metrics package for guiPymoo.

This package contains local metric implementations that can be
automatically discovered by the plugin system.

To add a new metric:
1. Create a .py file in this directory
2. Implement a callable that receives a front and returns a float
3. Export via get_metrics(), METRICS dict, or as a function
"""

__all__: list[str] = []
