
# Researcher

## Overview

This repository contains an implementation of a Research Agent designed for retrieving News & supporting research papers related to factors that impact
the insurance and the Reinsurance business.

## Features

#### 1.  Web Search : Agent searches for News related to Insurance and Reinsurance Business for a given time period, default being one month.

#### 2.  LLM Based Filtering : The retrieved news results are filtered based on their relevance to the task and appropriate tags are generated against each relevant news item.

#### 3.  LLM Based Dynamic Query Generation : Based on the filtered news results,dynamic queries are generated by the LLM to lookup relevant Research Papers.

#### 4.  LLM Based Filtering : The retrieved Research Papers are filtered based on their relevancy to the News Items and the respective URLs are extracted

#### 5.  Research Paper Extraction : The extracted URLs are then passed to a Search API for extracting the relevant content from the papers.

#### 6.  Output Parsing and Save to File : The outputs from the Research extract and the News items are parsed according to predefined readable formats and saved to a json file, which is then used as a database

#### The Web app then uses the tags from the news and research results and the associated data to render the UI. On Load the Web Page will contain tags and a default tag selected and the associated results. 

## Files

1. requirements.txt : Dependencies to be installed
2. .env : Contains the API keys, populate your API keys for OpenAI and Tavily
3. config.conf : Contains the variables to be used throughout the app
4. news_extractor.py : Contains the backend functions to be run 
5. app.py : Contains the front end logic and components
6. database.json : Contains the pre populated search results for default view

## Installation

  ### Prerequisites
   - Create Conda environment with Python 3.11.11
   - pip package manager
  
  ### Install the required dependencies 
  - pip install -r requirements.txt

## Run App

   - download repository
   - activate conda environment
   - update .env file with OpenAI and Tavily Search APIs
   - run command streamlit run app.py
