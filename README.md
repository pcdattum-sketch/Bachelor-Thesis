# Bachelor-Thesis

# Assessing the Capabilities of Sentinel-2 Multispectral Imagery for Soil Organic Carbon Predictions in German Croplands 

## Project Overview
This repository contains the code for a Bachelor thesis project: "Assessing the Capabilities of Sentinel-2 Multispectral Imagery for Soil Organic Carbon Predictions in German Croplands" The thesis was carried out by Chi Dat Pham and supervised by Prof. Dr. Dominik Grimm and Dr.Nikita Genze.

The study tested whether Sentinel-2 satellite imagery can predict topsoil soil organic carbon (SOC) content in German agricultural land as a lower-cost alternative to laboratory sampling. Soil organic carbon is measured through the LUCAS 2018 Topsoil survey, and spectral, terrain, and climate covariates are extracted at each sampling point through Google Earth Engine.

## Purpose
Soil organic carbon is a key indicator of soil quality and a factor in agricultural productivity and climate mitigation, but direct laboratory measurement at scale is slow and expensive. This project evaluated whether Sentinel-2 reflectance, combined with terrain and climate covariates, carries enough signal to estimate SOC content without lab analysis.

The pipeline extracted S2 bare-soil reflectance at LUCAS 2018 sampling points for Germany agriculture lands, five feature sets of increasing complexity, ranging from raw spectral bands to environmental features like typography aspects like elevation, aspect and slope, climatic like precipitation and temperature, and soil types like organic soil classification, clay content and available water capacity. A baseline set using soil chemical properties like nitrogen, phospohorus, potassium, pH level and electrical conductivity was created with the aim to compare features that rely on lab works versus those that don't.

This repository contains the code to the Machine Learning algorithms employed in this study. Methods such as L1-regularized linear regression, Random Forest, and Gradient Boosting were implemented and tuned. 

Model validation uses a stratified train/test split with the test set held out until final evaluation. Lastly, models are scored on R², RMSE, RPD, and RPIQ.

## Repository structure 
- predictor/: Implementation of the ML models used in the study and computing metrics 
- features_extraction/: Specific runs responsible for extracting features used in the study
- scripts/: A collection of Python codes used throughout the project, including data acquisition, preprocessing, and model evaluation tasks

