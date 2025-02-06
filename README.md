### Open Universiteit Capita Selecta - Generative AI - Assignment 3

# 🛰️ Remote Sensing Visual Question Answering 🛰️

This is our repository for assignment 3 for the capita selecta course in AI at Open Universiteit  
Please visit https://ou-capitaselecta.streamlit.app/ for a demonstration of our application.

In order to reproduce the results:
- Download the BigEarthNet-S2 dataset from https://bigearth.net/
- Use BigEarthNetPreProcessing from utils to create 3D and 6D equivalents of the data
- Generate questions and answers for each of the images using QuestionGenerator 
- Use model specific preprocessers from utils to prepare the data for each of our models
- Use the provided training processes to construct, train and evaluate the models 
