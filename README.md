🚧 **Work in progress** - Early development stage 🚧


# Deep Final Project
Neural Network to Predict Stock Prices

## Overview
This project uses financial news data from the Massive (formerly polygon.io) API to train a neural network for stock price prediction.

## Massive News API Setup

### Prerequisites
- Python 3.7 or higher
- A Massive API key (get one at https://massive.com )

### Installation

1. **Install required packages:**
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate deep_final_project
   ```

2. **Set up your API key:**
   
   Option A: Using environment variable (recommended)
   ```bash
   # On Windows (PowerShell)
   $env:MASSIVE_API_KEY="your_api_key_here"
   
   # On Windows (Command Prompt)
   set MASSIVE_API_KEY=your_api_key_here
   
   # On Linux/Mac
   export MASSIVE_API_KEY="your_api_key_here"
   ```

# Secure PyTorch (Required for Transformers 2025/2026)
torch>=2.8.0
torchvision
torchaudio

# Transformers Compatibility Bridge
tensorflow==2.15.0
keras==2.15.0
tf-keras==2.15.0

# NLP & Processing
transformers
sentence-transformers
pandas
numpy
tqdm
requests
matplotlib
```
