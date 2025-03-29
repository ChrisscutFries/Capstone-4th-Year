#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 02:27:28 2024

@author: mareyrey
"""

import joblib

# Load the saved model and vectorizer
model = joblib.load("optimized_ticket_classifier.pkl")
vectorizer = joblib.load("optimized_tfidf_vectorizer.pkl")

# Test with a sample summary
summary = "process running for over 600 hours"
summary_vec = vectorizer.transform([summary])
predicted_category = model.predict(summary_vec)[0]

print(f"Predicted Category: {predicted_category}")