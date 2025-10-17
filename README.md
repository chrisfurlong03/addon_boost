# Trip Boost
TripBoost uses an XGBoost model to rank addons for a flight booking with different discount levels by probability of purchasing.

The intention is to create a reccomendation system that allows for a simplified UX when booking. The left image is a screenshot of a recent step in the booking process for an actual flight and it displays over 10 bundle options. This is potentially overwhelming for users and may make purchasing decisions difficult. As a result, users might be more likely to disengage from purchasing an addon. 

>In order to improve the chance of purchasing, I propose a system that suggests fewer options that are most aligned to the user's specific context. The sketch on the right shows a potential UI for this new system. It displays only two options, but this allows for richer content like images to be displayed (improving product understanding). 

The **XGBoost model also tries to identify if discounts can be offered in order to improve the probability of purchase** and these discounts can be highlighted in this new UI.

![Alt text](images/Frame%202%20from%20Figma.png)

<a target="_blank" href="https://colab.research.google.com/github/chrisfurlong03/addon_boost/blob/main/Add_on_Bundling_Modeling.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## The XGBoost Model

## Serving the Model

## Training Outcomes

## Known Issues

## Potential Improvements