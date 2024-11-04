# Lab-Scripts
While working in biology several statistical and procedural processes are complicated with finding resources. This repository is meant to store and share easy-to-use tools for biologists.

# Violin Plot will take data stored in an Excel file in the same folder as the script. As long as the data is structured as follows:
  A           B
  Groups      Variable_Measurements
  Group A     1
  Group A     1.1
  Group A     1.5
  Group B     6.1
  Group B     6.8
  Group B     6.2

This will create a violin plot which can then be stored.
This graphing method does not exist in Excel so this was made as an easy alternative for a quick graphing solution.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

# Levene's Test

This test is done to evaluate the variance of the two groups and compare. This works as a hypothesis test and when the variance does no prove to be significantly different you'll receive a p-value of 0.05 < and when the variance is significantly different you'll see that p-value = < 0.05. This is valuable as different t-tests have different assumptions and one of them is the equivalence of variance (homoscedasticity). Equal variance parametric t-tests are known as Student's t-tests and parametric unequal variance (heteroscedastic) t-tests are known as Welch's t-tests.

Welch's t-tests are best for both unequal sample sizes and unequal variance however if the variance is equal then you should proceed with a Student's t-test. But fear not, the script will tell you the recommended t-test to perform. Some software will give you the name of the t-test others won't such as Excel. In Excel to perform the respective t-tests use:
  =t.test(range1, range2, 2, 2 (homoscedastic))
    This is a two-tailed Student's t-test
  =t.test(range1, range2, 2, 3 (heteoscedastic))
    This is a two-tailed Welch's t-test
Be very hesitant to use either a one-tailed or paired t-test. They both have valuable places in statistics however in most circumstances in biology these will not be appropriate to use. A paired t-test will require the comparison of the same individual, person, place, population, etc. before and after a change in conditions. A good example in molecular biology of where this would be appropriate is in an experiment measuring the growth of fungus in a race tube. The first measurement is done by measuring the distance between conidiation bands in standard conditions and then measuring that same growth rate measurement after being moved to 20-degree conditions. You'd collect these same measurements from multiple different replications of the experiment then you'd perform a t-test with 25-degree conditions as group 1 or control then your second conditions as group 2.
Excel procedure:
  =t.test(group1 (25-degree), group2 (20-degree), 2 (two-tailed), 1(paired))
A one-tailed t-test should only be done in circumstances where you hypothesize a move only in one direction. In my case, I give drugs to fungus to measure changes in core-clock protein expression. Since I know very little of how these compounds impact all of my organisms' biological systems every hypothesis needs to be tailored to the fact that much is unknown. I could foresee that if an increase in amplitude is observed for the light-sensitive element of neurospora's clock then any drug that increases expression of this protein would improve sensitivity to light compared to a control. However this rides on several assumptions including... no additional effects aside from core-clock interaction, LOV domain concentration dependence for changes in light sensitivity, and no antagonistic effects being produced by these drugs. Even though I could theory-craft a hypothesis for how light sensitivity should only move in one direction as long as the core-clock expression is amplified this is a flimsy hypothesis that cannot support a one-tailed t-test. 

A good example of a one-tailed t-test would be a trial on the best exercise to illicit protein synthesis. If you take untrained people and begin them on any form of training they will have heightened protein synthesis to recover. This has been observed routinely and would support a one-tailed t-test. In a test of untrained people getting several different forms of training will increase protein synthesis but to what extent is unknown. Since this would rely on a measurement of the same individuals before and after a change in condition (method of training) then the way to measure this in Excel would be a one-tailed paired t-test:
  =t.test(group1 (protein synthesis before training), group2 (protein synthesis after training), 1 (one-tailed), 1 (paired))
  
This interactive program once executed will require you to input the number of groups, labels, and comma-separated data. After successfully inputting these values you will receive p-values for your Levene's Tests and a recommendation for which type of t-test to use for your data. You will have to evaluate if you need to do a one-tailed/ two-tailed and or if a paired t-test is required. Use the information provided earlier to best assess the test needing to be executed.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Automated Phase Response Curve processing for Mammalian and N. Crassa Data

