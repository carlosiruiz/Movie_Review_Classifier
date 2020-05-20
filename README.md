# Movie Review Classifier

Gabriela Lopez

Sources: IMDB & Rotten Tomatoes

## Table of Contents

1. [Introduction](#intro)
    1. [Business Case](#case)
    2. [Libraries Used](#lib)
    3. [File Index](#files)
    4. [Presentation Link](#pp)
2. [Data Collection](#data)
3. [Data Cleaning & EDA](#eda)
4. [Modeling](#model)
    1. [Baseline](#base)
    2. [Final](#final)
5. [Conclusion](#end)
    1. [Results](#res)
    2. [Recommendations](#rec)

# Introduction <a id='intro'></a>

## Business Case <a id='case'></a>

My goal with this project is to build a sentiment classifier on movie reviews scraped from Rotten Tomatoes. This is a common NLP task, and a good way to practice working with text data.

## Libraries Used <a id='lib'></a>

This project uses the following python libraries.

#### Data Collection
* BeautifulSoup
* requests
* time
* pandas

#### Cleaning and EDA

*

#### Modeling

*

#### Custom Functions

* matplotlib
* numpy
* pandas
* sklearn
* xgboost
* itertools

## File Index <a id='files'></a>

#### Images/
Stores visualizations

#### Notebooks_and_Code/
Stores jupyter notebooks and python files with custom functions.

**Data_Collection.ipynb**: This workbook scrapes IMDB to create a list of movies for a given genre, and then Rotten Tomatoes for reviews for those movies.

**Cleaning_and_EDA.ipynb**: This workbook cleans and preprocesses the collected data so it is ready for modeling.

**Modeling.ipynb**: This workbook contains the model creation.

**custom.py**: This file contains my custom functions for creating and evaluating classifiers.

## Presentation Link <a id='pp'></a>

# Data Collection <a id='data'></a>

First, I collected a list of the top 1,000 movies of the horror genre from IMDB. I scraped all top critic reviews for every movie on this list from Rotten Tomatoes. The reviews and their scores were put into a dataframe. 

# Data Cleaning & EDA <a id='eda'></a>

# Modeling <a id='model'></a>

## Baseline <a id='base'></a>

## Final <a id='final'></a>

# Conclusion <a id='end'></a>

## Results <a id='res'></a>

## Recommendations <a id='rec'></a>