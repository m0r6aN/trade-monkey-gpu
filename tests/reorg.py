#!/usr/bin/env python3

import os
import sys

# Change to the correct directory
os.chdir('D:/Repos/trade-monkey-lite/trademonkey-lite')

# Add src to Python path
sys.path.append('src')

from backtesting.gpu_accelerated_backtester import run_gpu_christmas_special
import asyncio
asyncio.run(run_gpu_christmas_special())
