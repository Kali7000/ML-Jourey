# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:44:14 2026

@author: Kali
"""

import math

# Data: (Hours Studied, Passed?)
# 1hr -> Fail, 2hr -> Fail, 4hr -> Pass, 5hr -> Pass
data = [(1, 0), (2, 0), (4, 1), (5, 1)]

# 1. Initialize
w = 0.0
b = 0.0
lr = 0.1

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# 2. The Learning Loop
for epoch in range(1000):
    for x, y_actual in data:
        # Step A: Predict (Linear + Sigmoid)
        z = w * x + b
        y_pred = sigmoid(z) #why is the z value going through the sigmoid function and how does that help with classification
        
        # Step B: Calculate Error 
        # (The math simplifies beautifully here!)
        error = y_pred - y_actual
        
        # Step C: Update (Same "Nudge" logic as before)
        w = w - (lr * error * x)
        b = b - (lr * error)

# 3. Test the Bouncer
test_hours = 2
probability = sigmoid(w * test_hours + b)
prediction = 1 if probability >= 0.5 else 0

print(f"For {test_hours} hours:")
print(f" - Probability of passing: {probability:.2%}")
print(f" - Final Decision: {'PASS' if prediction == 1 else 'FAIL'}")