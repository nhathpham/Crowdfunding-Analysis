# Crowdfunding-Analysis
Predictive Analysis of Success and Failure on Crowdfunding Platforms Kickstarter and Indiegogo

<h2>I/ Abstract: </h2>
  In this project, I will analyze data on creative campaigns hosted on two popular crowdfunding sites, Kickstarter.com and Indiegogo.com, from 2009-2021, and use machine learning models for classification to predict whether a campaign will succeed or not. The project will also assess the impact of Covid-19 on donations to campaigns and compare indicators of success and failure between the two platforms.<br />

<h2>II/ Introduction</h2>
★ Problem: <br />
Crowdfunding has enabled creators, entrepreneurs, and investors around the world to start new businesses, fund initiatives, raise money, and build communities. It is an expanding industry with ~10 billion USD transaction value generated in 2021 globally (Fig.1)1. However, not all crowdfunding efforts reach their desired goals. Why do some projects succeed, while others fail?<br />
   As of 2021, there are over 1400 crowdfunding organizations in the US2. Currently, the three largest sites are Kickstarter.com, Indiegogo.com, and Crowsupply.com 3. A lot of factors contribute to the success or failure of a campaign on such platforms. Being able to assess whether or not a platform is a good funding option will help creators decide which one(s) to use to maximize their likelihood of success.<br />
   
![5010](https://user-images.githubusercontent.com/87089936/143928772-6f16a695-031f-4e25-bfa9-a3bdda940b93.png)<br />

★	Datasets:<br />
  Two datasets are downloaded from the web scraping site WebRobots.io 4, 5. They contain data on all projects hosted on Kickstarter.com between April 2009 and September 2021*, and on Indiegogo between April 2010 and September 2021*. I will refer to documentation from WebRobots.io and a similar Kickstarter dataset on Kaggle.com for description of attributes 6.<br />

★	Objectives:<br />
The goal of the project is to construct machine learning models for classification to determine important factors that had a positive effect and negative effect on the amount of money received and success rates of campaigns on each platform. A comparison of success/ failure indicators between two platforms will potentially inform creators of a better funding option. The project will also attempt to assess the impact of Covid-19 on campaign creation and outcome using descriptive analysis.<br />

<h2>III/ Preliminary and Future Work:</h2>
★	Anticipation:<br />
   Based on differences in the two companies’ business models, a major one being that Kickstarter only supports fixed funding (all or nothing system) while Indiegogo allows both fixed and flexible funding (creators can access partially raised funds) 7, I anticipate Indiegogo will have higher success rates than Kickstarter. Based on initial visualization of the Kickstarter data  (Fig.2), it seems like projects with smaller funding goals and shorter duration from creation to launch will be more likely to succeed on this site.<br />

![5010 1](https://user-images.githubusercontent.com/87089936/143928771-5c3bb8fd-55df-4c4b-85ad-8279b8195734.png)<br />
★	Methods:<br />
     I will be using Python’s pandas, numpy, seaborn, and matplotlib packages to perform data tidying, pre-processing, and data visualization. I will apply two machine learning models for classification to the data to classify successes and failures, including logistic regression and random forest, using Python’s sklearn package.<br />
