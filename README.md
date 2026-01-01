# Keystroke Dynamics: Behavioral User Identification

## Overview  
This project studies whether people can be identified or verified using only how they type, based on keystroke timing data. The text being typed is the same for everyone, so the focus is only on typing behavior, not content.

## Data  
The project uses the CMU Keystroke Dynamics Benchmark dataset. It contains data from 51 subjects who repeatedly typed the same fixed password across multiple sessions. Each sample includes timing features such as how long keys are held and the time between key presses.

## Methods  
The project includes the following steps:
- Session aware train and test splits to avoid data leakage  
- Multiclass user identification using logistic regression  
- Cross validation to check performance stability  
- Per-user accuracy analysis to understand differences between individuals  
- Within-user variability analysis of typing behavior  
- A binary verification task that predicts whether two samples come from the same user  

## Results  
The logistic regression model achieved about 78 percent accuracy across 51 users, compared to a random chance level of about 2 percent. Performance was consistent across different train and test splits, but accuracy varied a lot between users. Some users were easy to identify, while others were often confused with others.

Within-user variance alone did not explain why some users were easier to identify than others. The verification task performed well, with a ROC AUC of about 0.87, showing that typing behavior contains useful identity information.

A Random Forest model was also tested for the identification task. It improved accuracy to about 86 percent with similar stability. This suggests that typing identity is not purely linear and that interactions between timing features matter.

Feature importance was analyzed across multiple Random Forest models. Important features were consistent across splits and mainly involved modifier keys like Shift and Return, letter hold times, and transitions between keys. Features related to repetition or session order had very little impact. This shows that identity information comes from fine motor behavior rather than simple timing patterns.

To better explain differences in performance, I included a separability metric. This metric compares how far a user is from other users relative to how spread out their own samples are. Separability showed a positive relationship with identification accuracy. Users whose typing patterns were more distinct from others tended to be identified more accurately, even if their typing was not perfectly consistent.

### Advanced Model: XGBoost  
A gradient boosted tree model was tested to see if a more complex model could further improve identification accuracy. XGBoost achieved similar performance to Random Forest, around 86 percent accuracy, with comparable stability. This suggests that most useful structure in the data is already captured by tree based models, and boosting provides limited additional benefit.

## Key Insight  
Typing identity depends more on how different a person’s typing patterns are from others than on how consistent their typing is across attempts.

## Limitations  
- Only fixed text typing was studied  
- Data was collected in a controlled environment  
- Long term changes in typing behavior were not analyzed  

## Ethical Note  
Keystroke dynamics can be used as a behavioral biometric, which could raise privacy concerns if applied without a user’s knowledge or consent.
