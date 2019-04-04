README - RL_MDD_all

Alex Filipowicz - March 21st, 2019

This folder contains the data from a 2-armed bandit task administered to Healthy Controls and patients with Major Depressive Disorder (MDD)

During this task subjects saw two fractals on a screen and were instructed to choose the fractal they thought would maximize their chance of getting a reward. At any point throughout the task, one of the fractals had a higher chance of giving the subject a reward (reward probability of .75) while the other had a lower probability (.25). Every 30 trials, the "rich" fractal switched - subjects were not made aware of these switches and had to infer from the task feedback which of the fractals was most rewarding.

On each trial the side on which the fractal appeared (left or right) was pseudo randomized. This was done to ensure that participants didn't just choose to repeat one action all the time, but were really thinking about the fractals.

Overall each subject participated in two conditions: a reward condition or a punishment condition. In the reward condition, subjects started with 0 points and got +1 points for every trial on which they got a reward. In the punishment condition, subjects started with a certain amount of points, and lost -1 every time they chose a fractal that punished them.

All of the data from this task are in the RL_MDD_All.csv file. It's structured as follows:

Column 1: Row index (can be ignored)
Subject: Subject Identifier
SubNum: Another subject identifier used for other reasons
Group: Subject Group (Healthy Control or MDD)
SubChoice: Fractal chosen by the subject (1 or 2)
SubAction: Button pressed by the subject (1 for left 2 for right)
RichFrac: Best fractal to pick to maximize reward (in reward condition)/minimize loss (in punish condition)
Condition: Reward or Punishment condition
RichFracChoice: Whether the subject chose the best fractal (even if it wasn't rewarded)
Fractal1_Side: Side on which fractal 1 appeared (1 for left 2 for right)
Fractal2_Side: Side on which fractal 2 appeared (1 for left 2 for right)
Fractal_1_Reward: Would fractal 1 have been rewarded if it had been chosen on that trial
Fractal_2_Reward: Would fractal 2 have been rewarded if it had been chosen on that trial
TSCP: Trial since change in the rich fractal
Reward: Whether or not the trial was rewarded (in the reward condition - 1 for reward 0 for no reward) or whether or not the trial resulted in a loss (for the punishment condition 1 for no loss 0 for loss)

FOR INFORMAITION BOTTLENECK:

As a first pass, here are what I think the features should look like:

SubChoice, Reward, RichFrac - these are everything needed for the task itself

Then we can think about adding the actions.