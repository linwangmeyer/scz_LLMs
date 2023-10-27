rm(list=ls())

## -- Set function defaults:
filter <- dplyr::filter
group_by <- dplyr::group_by
summarize <- dplyr::summarize
select <- dplyr::select
rename <- dplyr::rename

## -- Set working directory and install packages:
if(!require(pacman)) {install.packages("pacman"); require(pacman)}

p_load("ggplot2", "rstudioapi", "tidyverse", "lme4", "lmerTest", 
       "car", "patchwork", "afex", "yarrr", "hypr", "MASS", 
       "emmeans", "udpipe")

#setwd(dirname(getActiveDocumentContext()$path))    ## sets dir to R script path
setwd("/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/")


#---------------------------------------------#
#prepare data
#---------------------------------------------#
data <- read.csv(file = 'TOPSY_TwoGroups.csv')
df <- data[,c('ID','PatientCat','Gender','AgeScan1','stim','TLI_DISORG','n_sentence','entropyApproximate')]
df2 <- df %>% select(ID, PatientCat, TLI_DISORG, n_sentence, stim,  entropyApproximate) %>% 
  mutate(ID = as.factor(ID),
         PatientCat = as.factor(PatientCat),
         stim = as.factor(stim))

#visualize data
x1 <- df$entropyApproximate[df2$PatientCat==1]
x2 <- df$entropyApproximate[df2$PatientCat==2]
vioplot(x1, x2, names=c("ctrl", "scz"), colors=c("blue", "red"))


############################run models###################
m1 <- lmer(entropyApproximate ~ PatientCat + stim + n_sentence + TLI_DISORG + (1 | ID), data = df2)
m2 <- lmer(entropyApproximate ~ PatientCat + n_sentence + TLI_DISORG + (1 | ID), data = df2)
m3 <- lmer(entropyApproximate ~ PatientCat + stim + TLI_DISORG + (1 | ID), data = df2)
m4 <- lmer(entropyApproximate ~ stim + n_sentence + TLI_DISORG + (1 | ID), data = df2)
anova(m1, m4)

#check the influence of TLI_DISORG
df3 <- df2[!is.na(df2$TLI_DISORG), ]
m4 <- lmer(entropyApproximate ~ PatientCat + stim + n_sentence + TLI_DISORG + (1 | ID), data = df3)
m5 <- lmer(entropyApproximate ~ PatientCat + stim + n_sentence + (1 | ID), data = df3)
anova(m4, m5)

m1 <- lmer(entropyApproximate ~ PatientCat + stim + n_sentence + TLI_DISORG + (1 | ID), data = df2)
summary(m1)


