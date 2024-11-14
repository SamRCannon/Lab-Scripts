# Lab-Scripts
While working in biology several statistical and procedural processes are complicated with finding resources. This repository is meant to store and share easy-to-use tools for biologists.

Note: Each script is made to be operated by people with minimal experience in Python so redundant imports like sys are rerun every time to ensure any first-time user does not have to troubleshoot errors. To make this easier the function installx() is a header for all scripts to automatically handle installations and imports.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

# Violin Plot will take data stored in an Excel file in the same folder as the script. As long as the data is structured as follows:

An Excel file needs to be generated with column A starting with a label followed by group ID for sample 1. Column B starts with a label and is then followed by the first value for sample 1. Continue this process with every replicate before saving. (see the example file for additional guidance)

Double-click the script while it's in the same folder as the Excel file and it will take about 5 seconds to generate.

This will create a violin plot which can then be stored.
This graphing method does not exist in Excel so this was made as an easy alternative for a quick graphing solution.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

# Stats Test

This test is done to evaluate the variance of the two groups and compare. This works as a hypothesis test and when the variance does no prove to be significantly different you'll receive a p-value of 0.05 < and when the variance is significantly different you'll see that p-value = < 0.05. This is valuable as different t-tests have different assumptions and one of them is the equivalence of variance (homoscedasticity). Equal variance parametric t-tests are known as Student's t-tests and parametric unequal variance (heteroscedastic) t-tests are known as Welch's t-tests.

Welch's t-tests are best for both unequal sample sizes and unequal variance however if the variance is equal then you should proceed with a Student's t-test. But fear not, the script will tell you the recommended t-test to perform. Some software will give you the name of the t-test others won't such as Excel. In Excel to perform the respective t-tests use:

  =t.test(range1, range2, 2, 2 (homoscedastic))
    This is a two-tailed Student's t-test

  =t.test(range1, range2, 2, 3 (heteoscedastic))
    This is a two-tailed Welch's t-test

Be very hesitant to use either a one-tailed or paired t-test. They both have valuable places in statistics however in most circumstances in biology these will not be appropriate to use. A paired t-test will require comparing the same individual, person, place, population, etc. before and after a change in conditions. A good example in molecular biology of where this would be appropriate is in an experiment measuring fungus growth in a race tube. The first measurement is done by measuring the distance between conidiation bands in standard conditions and then measuring that same growth rate measurement after being moved to 20-degree conditions. You'd collect these same measurements from multiple different replications of the experiment, then perform a t-test with 25-degree conditions as group 1 or control, then your second conditions as group 2.

Excel procedure:
  =t.test(group1 (25-degree), group2 (20-degree), 2 (two-tailed), 1(paired))

A one-tailed t-test should only be done in circumstances where you hypothesize a move only in one direction. In my case, I give drugs to fungus to measure changes in core-clock protein expression. Since I know very little of how these compounds impact all of my organisms' biological systems every hypothesis needs to be tailored to the fact that much is unknown. I could foresee that if an increase in amplitude is observed for the light-sensitive element of neurospora's clock then any drug that increases expression of this protein would improve sensitivity to light compared to a control. However this rides on several assumptions including... no additional effects aside from core-clock interaction, LOV domain concentration dependence for changes in light sensitivity, and no antagonistic effects being produced by these drugs. Even though I could theory-craft a hypothesis for how light sensitivity should only move in one direction as long as the core-clock expression is amplified this is a flimsy hypothesis that cannot support a one-tailed t-test. 

A good example of a one-tailed t-test would be a trial on the best exercise to illicit protein synthesis. If you take untrained people and begin them on any form of training they will have heightened protein synthesis to recover. This has been observed routinely and would support a one-tailed t-test. In a test of untrained people getting several different forms of training will increase protein synthesis but to what extent is unknown. Since this would rely on a measurement of the same individuals before and after a change in condition (method of training) then the way to measure this in Excel would be a one-tailed 

paired t-test:
  =t.test(group1 (protein synthesis before training), group2 (protein synthesis after training), 1 (one-tailed), 1 (paired))
  
This interactive program once executed will require you to input the number of groups, labels, and comma-separated data. After successfully inputting these values you will receive p-values for your Levene's Tests and a recommendation for which type of t-test to use for your data. You will have to evaluate if you need to do a one-tailed/ two-tailed and or if a paired t-test is required. Use the information provided earlier to best assess the test needing to be executed.

Additional Tests Performed

Bonferroni Adjustment/Correction: A Bonferroni Correction is a method used that attempts to correct type 1 errors. It's best used for multiple hypothesis testing but can be used at any time. It works by multiplying the p-value by the number of tests performed. The adjusted p-value will be larger than the p-value produced by a simple t-test and thus requires stronger data to meet the level of significance.

Tukey Test:
A Tukey test is like a more comprehensive ANOVA. They rely on the same assumptions however a Tukey test will compare all possible pairs of groups provided. This means you'll be able to see if treatment groups vary significantly from each other as well as how all treatments compare to control.

Dunnett's Test:
This test will evaluate treatment group means and compare them against a control mean to validate significance. Both Dunnett's and a Tukey HSD test are done after an ANOVA. This is done as an ANOVA provides an answer on whether the mean difference between groups is significant however these tests are far more illustrative on where the significant differences lie.

Anita Nanda, Dr. Bibhuti Bhusan Mohapatra, Abikesh Prasada Kumar Mahapatra, Abiresh Prasad Kumar Mahapatra, Abinash Prasad Kumar Mahapatra. Multiple comparison test by Tukey’s honestly significant difference (HSD): Do the confident level control type I error. Int J Stat Appl Math 2021;6(1):59-65. DOI: 10.22271/maths.2021.v6.i1a.636

### Stats Test Output

There will be three files saved in the directory where the file is executed.
  1. Levene_and_t_test_Results_{Current date}
  2. Dunnett_Test_Results_{Current date}
  3. Tukey_HSD_Results_{Current date}

### Interpretting output:

The Levene and t-test file will give a recommendation for t-test to used based on two seperate factors. The first is on the basis of the Levene's test p-value. If the p-value is less than 0.05 then a Welch's t-test is recommended since the groups are determined to be heteroscedastic. In cases where the Levene p-value exceeds 0.05 a Student's t-test is recommended. The second factor is based on difference is sample sizes. The reason sample size matters is that it can very easily mask unequal variance especially in small sample sizes. If the difference in sample size is observed to be greater than 40% then a recommendation to use a welch's is presented in the file.

Regardless of recommendation both t-tests and their respective Bonferroni adjustments are provided for you to use at your discression. 

Both the Dunnett and Tukey tests are stored independently. With the Dunnett providing only p-values and the Tukey test giving far more with the "Reject" referring to whether the hypothesis of no significance is accepted or rejected. 

-------------------------------------------------------------------------------------------------------------------------------------------------------------

# Automated Phase Response Curve Processing for Mammalian and N. Crassa Data

