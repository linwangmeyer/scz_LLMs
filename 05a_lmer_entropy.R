rm(list=ls())
install.packages("vioplot")
library(vioplot)


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
df <- data[,c('ID','PatientCat','Gender','AgeScan1','SES','PANSS.Pos','stim','TLI_DISORG','num_all_words','num_content_words','num_repeated_words','entropyApproximate')]
df <- df[df$stim != 'Picture4', ]

df2 <- df %>% 
  mutate(ID = as.factor(ID),
         Gender = as.factor(Gender),
         PatientCat = as.factor(PatientCat),
         stim = as.factor(stim))

df2 <- df2[!is.na(df2$entropyApproximate), ]

#visualize data
x1 <- df2$entropyApproximate[df2$PatientCat==1]
x2 <- df2$entropyApproximate[df2$PatientCat==2]
vioplot(x1, x2, names=c("ctrl", "scz"), colors=c("blue", "red"))


############################run models###################
#-----------------------------------------------------------#
# run models: average across stim items --> linear regression model
# individual reference level; include all predictors
# different subsets of participants, depending on the included variables
#-----------------------------------------------------------#
data <- read.csv(file = 'TOPSY_TwoGroups.csv')
df <- data[,c('ID','PatientCat','Gender','AgeScan1','SES','PANSS.Pos','Trails.B', 'Category.Fluency..animals.','DSST_Writen','DSST_Oral','TLI_DISORG','num_all_words','num_content_words','num_repeated_words','n_sentence','stim','entropyApproximate')]
df <- df[df$stim != 'Picture4', ]
df <- df[!is.na(df$entropyApproximate),]

df$DSST <- (df$DSST_Oral + df$DSST_Writen)/2

df2 <- df %>%
  group_by(ID) %>%
  summarise(topic_mean = mean(as.numeric(entropyApproximate), na.rm = TRUE),
            nword_mean = mean(as.numeric(num_all_words), na.rm = TRUE),
            ncontent_mean = mean(as.numeric(num_content_words), na.rm = TRUE),
            nrepeated_mean = mean(as.numeric(num_repeated_words), na.rm = TRUE),
            nsen_mean = mean(as.numeric(n_sentence), na.rm = TRUE)) %>%
  ungroup()

# Extracting other columns from the original dataframe for df3
other_columns <- df %>%
  select(ID, PatientCat, Gender, AgeScan1, SES, PANSS.Pos, TLI_DISORG, DSST, Trails.B, Category.Fluency..animals.) %>%
  distinct()

# Merging all the columns together based on 'ID'
df3 <- merge(df2, other_columns, by = "ID", all = TRUE)

#----------------------------
# get data containing all control demographic variables excluding SES: 34 HC + 70 FEP
df4 <- df3 %>%
  select(ID, PatientCat, Gender, AgeScan1, TLI_DISORG, nsen_mean, nword_mean, ncontent_mean, nrepeated_mean, topic_mean) %>% 
  mutate(across(c(ID, PatientCat, Gender), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values

sum(df4$PatientCat==1) #HC: 34
sum(df4$PatientCat==2) #FEP: 70

#----------------------------
# get data containing all control demographic variables including SES: 33 HC + 60 FEP
df5 <- df3 %>%
  select(ID, PatientCat, Gender, AgeScan1, SES, TLI_DISORG, nsen_mean, nword_mean, ncontent_mean, nrepeated_mean, topic_mean) %>% 
  mutate(across(c(ID, PatientCat, Gender), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values

sum(df5$PatientCat==1) #HC: 33
sum(df5$PatientCat==2) #FEP: 60

#----------------------------
# get data containing all control variables, SES + cognitive functions: 29 HC + 42 FEP
df6 <- df3 %>%
  select(ID, PatientCat, Gender, AgeScan1, SES, DSST, Trails.B, Category.Fluency..animals., TLI_DISORG, nsen_mean, nword_mean, ncontent_mean, nrepeated_mean, topic_mean) %>% 
  mutate(across(c(ID, PatientCat, Gender), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values

sum(df6$PatientCat==1) #HC: 29
sum(df6$PatientCat==2) #FEP: 42

#----------------------------
# get data containing all variables, including SES, PANSS.Pos and cognitive functions: 24 HC + 40 FEP
df7 <- df3 %>%
  select(ID, PatientCat, Gender, AgeScan1, SES, PANSS.Pos, DSST, Trails.B, Category.Fluency..animals., TLI_DISORG, nsen_mean, nword_mean, ncontent_mean, nrepeated_mean, topic_mean) %>% 
  mutate(across(c(ID, PatientCat, Gender), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values

sum(df7$PatientCat==1) #HC: 24
sum(df7$PatientCat==2) #FEP: 40


#---------- continuous effect
# all participants
m_grand4 = lm(topic_mean ~ TLI_DISORG + nword_mean + Gender + AgeScan1, data = df4) 
summary(m_grand4)

m_grand4b = lm(topic_mean ~ TLI_DISORG + nsen_mean + Gender + AgeScan1, data = df4) 
summary(m_grand4b)

m_grand4c = lm(topic_mean ~ TLI_DISORG + Gender + AgeScan1, data = df4) 
summary(m_grand4c)

# only participants with SES
m_grand5 = lm(topic_mean ~ TLI_DISORG + nword_mean + Gender + AgeScan1 + SES, data = df5) 
summary(m_grand5)

m_grand5b = lm(topic_mean ~ TLI_DISORG + nsen_mean + Gender + AgeScan1 + SES, data = df5) 
summary(m_grand5b)

m_grand5c = lm(topic_mean ~ TLI_DISORG + Gender + AgeScan1 + SES, data = df5) 
summary(m_grand5c)

# only participants with both SES and cognitive measures
m_grand6 = lm(topic_mean ~ TLI_DISORG + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals. + nword_mean, data = df6) 
summary(m_grand6)

m_grand6b = lm(topic_mean ~ TLI_DISORG + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals. + nsen_mean, data = df6) 
summary(m_grand6b)

m_grand6c = lm(topic_mean ~ TLI_DISORG + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals., data = df6) 
summary(m_grand6c)

# only participants with SES, cognitive measures and PANSS.Pos: 24 HC + 40 FEP
m_grand7 = lm(topic_mean ~ TLI_DISORG + PANSS.Pos + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals. + nword_mean, data = df7) 
summary(m_grand7)

m_grand7b = lm(topic_mean ~ TLI_DISORG + PANSS.Pos + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals. + nsen_mean, data = df7) 
summary(m_grand7b)



#---------- group effect
# all participants
m_grand1 = lm(topic_mean ~ PatientCat + nword_mean + Gender + AgeScan1, data = df4) 
summary(m_grand1)

m_grand1b = lm(topic_mean ~ PatientCat + nsen_mean + Gender + AgeScan1, data = df4) 
summary(m_grand1b)

# only participants with SES
m_grand2 = lm(topic_mean ~ PatientCat + nword_mean + Gender + AgeScan1 + SES, data = df5) 
summary(m_grand2)

m_grand2b = lm(topic_mean ~ PatientCat + nsen_mean + Gender + AgeScan1 + SES, data = df5) 
summary(m_grand2b)

# only participants with both SES and cognitive measures
m_grand3 = lm(topic_mean ~ PatientCat + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals. + nword_mean, data = df6) 
summary(m_grand3)

m_grand3b = lm(topic_mean ~ PatientCat + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals. + nsen_mean, data = df6) 
summary(m_grand3b)



#---------------------------------------------#
# run models: take stim items as fixed effects
# individual reference level
# include all predictors
# 86 participants due to missing SES and PANSS.Pos
#---------------------------------------------#
data <- read.csv(file = 'TOPSY_TwoGroups.csv')
df <- data[,c('ID','PatientCat','Gender','AgeScan1','SES','PANSS.Pos','Trails.B', 'Category.Fluency..animals.','DSST_Writen','DSST_Oral','TLI_DISORG','num_all_words','num_content_words','num_repeated_words','n_sentence','stim','entropyApproximate')]
df <- df[df$stim != 'Picture4', ]
df <- df[!is.na(df$entropyApproximate),]

df$DSST <- (df$DSST_Oral + df$DSST_Writen)/2

df2 <- df %>%
  select(ID, PatientCat, Gender, AgeScan1, SES, PANSS.Pos, DSST, Trails.B, Category.Fluency..animals., TLI_DISORG, num_all_words, num_content_words, num_repeated_words, n_sentence, entropyApproximate, stim) %>% 
  mutate(across(c(ID, PatientCat, Gender), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values


#------------ model
m1 <- lmer(entropyApproximate ~ PatientCat + Gender + AgeScan1 + SES + stim + n_sentence  + (1 | ID), data = df2)
summary(m1)

m2 <- lmer(entropyApproximate ~ PatientCat + Gender + AgeScan1 + SES + stim + n_sentence  + (1 | ID), data = df2)
m3 <- lmer(entropyApproximate ~ stim + TLI_DISORG + n_sentence + (1 | ID), data = df2)
m4 <- lmer(entropyApproximate ~ stim + n_sentence + TLI_DISORG + (1 | ID), data = df2)
anova(m1, m2)


#check the influence of TLI_DISORG
df3 <- df2[!is.na(df2$TLI_DISORG), ]
m4 <- lmer(entropyApproximate ~ PatientCat + stim + n_sentence + TLI_DISORG + (1 | ID), data = df3)
m5 <- lmer(entropyApproximate ~ PatientCat + stim + n_sentence + (1 | ID), data = df3)
anova(m4, m5)

m1 <- lmer(entropyApproximate ~ PatientCat + stim + n_sentence + TLI_DISORG + (1 | ID), data = df2)
summary(m1)

# ---------------------
# main effects
myMat <- solve(t(matrix(c(1/3,1/3,1/3,-1,1,0,-1,0,1), nrow =3, ncol = 3)))
contrasts(df2$stim) <- myMat
design_matrix <- hypr()
cmat(design_matrix, add_intercept = FALSE) <- myMat
design_matrix

contrasts(df2$PatientCat) <- c(-.5, .5)
m5 <- lmer(entropyApproximate ~ PatientCat*TLI_DISORG + Gender + AgeScan1 + SES + stim + n_sentence  + (1 | ID), data = df2)
emm <- emmeans(m5, pairwise ~ TLI_DISORG | PatientCat, pbkrtest.limit = 14060)
pairs(emm, adjust = "tukey")
summary(m5)

#---------------------
# average across items
mean_values <- df2 %>%
  group_by(ID) %>%
  summarise(meanentropy = mean(entropyApproximate, na.rm = TRUE),
            nword_mean = mean(as.numeric(num_all_words), na.rm = TRUE),
            ncontent_mean = mean(as.numeric(num_content_words), na.rm = TRUE),
            nrepeated_mean = mean(as.numeric(num_repeated_words), na.rm = TRUE))
df3 <- left_join(df2, mean_values, by = "ID")

#df3$PatientCat <- contr.treatment(levels(df3$PatientCat))
#contrasts(df3$PatientCat) <- c(-.5, .5)

m6 <- lmer(entropyApproximate ~ PatientCat + TLI_DISORG + Gender + AgeScan1 + SES + nword_mean  + (1 | ID), data = df3)
emm <- emmeans(m6, pairwise ~  PatientCat, pbkrtest.limit = 14060)
pairs(emm, adjust = "tukey")
summary(m6)