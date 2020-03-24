See my notebook "yelp-reviews-project.ipynb" for now.

# Automated Crowdturfing
**Generating Realistic Reviews**

## Andrew Kin-Yip Chien
[Linkedin](https://www.linkedin.com/in/andrew-k-chien/) | [Github](https://github.com/kchien3) | [Slides](https://github.com/kchien3/yelp-reviews-project/blob/master/presentation/yelp_reviews_project-slides.pdf)

## Table of Contents

* [Background and Motivation](#background-and-motivation)
* [Data](#data)
  * [Description](#description)
* [Feature Engineering](#feature-engineering)
* [Exploration](#exploration)
* [Revenue Maximization Strategies](#revenue-maximization-strategies)
* [Conclusions](#conclusions)
* [Future Directions](#future-directions)
* [References](#references)

## Background and Motivation
[82%](www.brightlocal.com/research/local-consumer-review-survey) of consumers read online reviews for local businesses.  
[76%](www.brightlocal.com/research/local-consumer-review-survey) of consumers trust online reviews as much as recommendations from family and friends.  
A one-star increase in yelp rating leads to a [5-9%](https://www.hbs.edu/faculty/Pages/item.aspx?num=41233) increase in revenue.  
Having good online reviews is vital for business success. Because it can be difficult to implement organizational changes that lead to good reviews, like investing in decor, training and the product (eg. food), an easier route to good reviews is 'crowdturfing,' or hiring people to write good reviews for services they never received. This project explores automated crowdturfing -- training a neural network to generate positive reviews for businesses, as well as methods to detect generated reviews.

## Data
### Description
Yelp provides [over 15 million user reviews](https://www.yelp.com/dataset) on their website for anybody to explore. In this project, various subsets of reviews are used for different investigations, with the size of subsets varying due to computational considerations.

A neural network model was used to generate a corpus of review text consisting of approximately 2.8 million characters (500,000 words) and was used as part of the training set for building classifiers

## References
* Murphy, Rosie. “Local Consumer Review Survey: Online Reviews Statistics & Trends.” BrightLocal, 28 Jan. 2020, www.brightlocal.com/research/local-consumer-review-survey.
* Luca, Michael. "[Reviews, Reputation, and Revenue: The Case of Yelp.com](https://www.hbs.edu/faculty/Pages/item.aspx?num=41233)." Harvard Business School Working Paper, No. 12-016, September 2011. (Revised March 2016. Revise and resubmit at the American Economic Journal - Applied Economics.)
