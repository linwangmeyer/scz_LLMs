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
df <- data[,c('ID','PatientCat','Gender','AgeScan1','SES','PANSS.Pos','stim','TLI_DISORG','n_sentence','entropyApproximate')]
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
m1 <- lmer(entropyApproximate ~ PatientCat*TLI_DISORG + Gender + AgeScan1 + SES + stim + n_sentence  + (1 | ID), data = df2)
m2 <- lmer(entropyApproximate ~ PatientCat + TLI_DISORG + Gender + AgeScan1 + SES + stim + n_sentence  + (1 | ID), data = df2)
m3 <- lmer(entropyApproximate ~ PatientCat + stim + TLI_DISORG + (1 | ID), data = df2)
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
# average across tims
mean_values <- df2 %>%
  group_by(ID) %>%
  summarise(meanvalue = mean(entropyApproximate, na.rm = TRUE))
df3 <- left_join(df2, mean_values, by = "ID")

#df3$PatientCat <- contr.treatment(levels(df3$PatientCat))
#contrasts(df3$PatientCat) <- c(-.5, .5)

m6 <- lmer(entropyApproximate ~ PatientCat + TLI_DISORG + Gender + AgeScan1 + SES + n_sentence  + (1 | ID), data = df3)
emm <- emmeans(m6, pairwise ~  PatientCat, pbkrtest.limit = 14060)
pairs(emm, adjust = "tukey")
summary(m6)






#---------------------------------------------#
# run models: average across stim items
# individual reference level
# include all predictors
#---------------------------------------------#
data <- read.csv(file = 'TOPSY_TwoGroups.csv')
df <- data[,c('ID','PatientCat','Gender','AgeScan1','SES','PANSS.Pos','TLI_DISORG','n_sentence','stim','entropyApproximate')]
df <- df[df$stim != 'Picture4', ]
df <- df[!is.na(df$entropyApproximate),]

df2 <- df %>%
  group_by(ID) %>%
  summarise(topic_mean = mean(as.numeric(entropyApproximate), na.rm = TRUE),
            nsen_mean = mean(as.numeric(n_sentence), na.rm = TRUE)) %>%
  ungroup()

# Extracting other columns from the original dataframe for df3
other_columns <- df %>%
  select(ID, PatientCat, Gender, AgeScan1, SES, PANSS.Pos, TLI_DISORG) %>%
  distinct()

# Merging all the columns together based on 'ID'
df3 <- merge(df2, other_columns, by = "ID", all = TRUE)

df4 <- df3 %>%
  select(ID, PatientCat, Gender, AgeScan1, SES, PANSS.Pos, nsen_mean, TLI_DISORG, topic_mean) %>%
  mutate(across(c(ID, PatientCat, Gender), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values

# Calculate level frequencies and remove empty levels
id_summary <- table(df4$ID)
valid_levels <- names(id_summary[id_summary > 0])
df4 <- droplevels(df4[df4$ID %in% valid_levels, ])


#---------- group effect
# intercept being healthy controls, and n_1
m_grand1 = lmer(topic_mean ~ PatientCat + (1 | ID), data = df4) 
m_grand2 = lmer(topic_mean ~ PatientCat + (1 | ID) + nsen_mean + Gender + AgeScan1 + SES, data = df4) 
anova(m_grand1,m_grand2)
summary(m_grand1)
summary(m_grand2)

# for each word position, group effect
emm <- emmeans(m_grand3, pairwise ~ PatientCat | wordpos, pbkrtest.limit = 14060)
pairs(emm, adjust = "tukey")

# for each group, the word position effect
emm <- emmeans(m_grand3, pairwise ~ wordpos | PatientCat, pbkrtest.limit = 14060)
pairs(emm, adjust = "tukey")

#---------- continuous effect
m_grand4 = lmer(topic_mean ~ (1 | ID) + Gender + AgeScan1 + SES + nsen_mean + PANSS.Pos, data = df4) 
summary(m_grand4)

m_grand5 = lmer(topic_mean ~ (1 | ID) + Gender + AgeScan1 + SES + nsen_mean + TLI_DISORG, data = df4) 
summary(m_grand5)

m_grand6 = lm(topic_mean ~ TLI_DISORG + Gender + AgeScan1 + SES + nsen_mean, data = df4) 
summary(m_grand6)

#----
df4 <- df4 %>%
  mutate(
    PANSS_Pos_Z = scale(PANSS.Pos, center = TRUE, scale = TRUE),
    TLI_DISORG_Z = scale(TLI_DISORG, center = TRUE, scale = TRUE)
  ) %>%
  # Calculate comScore by computing the row-wise mean of PANSS_Pos_Z and TLI_DISORG_Z
  mutate(
    comScore = rowMeans(select(., c("PANSS_Pos_Z", "TLI_DISORG_Z")), na.rm = TRUE)
  )

m_grand6 = lmer(topic_mean ~ (1 | ID) + Gender + AgeScan1 + SES + nsen_mean + comScore, data = df4) 
summary(m_grand6)



#---------------------------------------------#
# run models: average across stim items
# individual reference level
# include only main predictors
#---------------------------------------------#
data <- read.csv(file = 'TOPSY_TwoGroups.csv')
df <- data[,c('ID','PatientCat','PANSS.Pos','TLI_DISORG','n_sentence','stim','entropyApproximate')]
df <- df[df$stim != 'Picture4', ]
df <- df[!is.na(df$entropyApproximate),]

df2 <- df %>%
  group_by(ID) %>%
  summarise(topic_mean = mean(as.numeric(entropyApproximate), na.rm = TRUE),
            nsen_mean = mean(as.numeric(n_sentence), na.rm = TRUE)) %>%
  ungroup()

# Extracting other columns from the original dataframe for df3
other_columns <- df %>%
  select(ID, PatientCat, PANSS.Pos, TLI_DISORG) %>%
  distinct()

# Merging all the columns together based on 'ID'
df3 <- merge(df2, other_columns, by = "ID", all = TRUE)

df4 <- df3 %>%
  select(ID, PatientCat, PANSS.Pos, TLI_DISORG, topic_mean) %>%
  mutate(across(c(ID, PatientCat), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values

# Calculate level frequencies and remove empty levels
id_summary <- table(df4$ID)
valid_levels <- names(id_summary[id_summary > 0])
df4 <- droplevels(df4[df4$ID %in% valid_levels, ])


#---------- group effect
# intercept being healthy controls, and n_1
m_grand1 = lmer(topic_mean ~ PatientCat + (1 | ID), data = df4) 
m_grand2 = lmer(topic_mean ~ PatientCat + (1 | ID) + nsen_mean, data = df4) 
anova(m_grand1,m_grand2)
summary(m_grand1)
summary(m_grand2)


#---------- continuous effect
m_grand4 = lmer(topic_mean ~ (1 | ID) + PANSS.Pos, data = df4) 
summary(m_grand4)

m_grand5 = lmer(topic_mean ~ (1 | ID) + nsen_mean + TLI_DISORG, data = df4) 
summary(m_grand5)


m_grand6 = lm(topic_mean ~ TLI_DISORG, data = df4) 
summary(m_grand6)


#----
df4 <- df4 %>%
  mutate(
    PANSS_Pos_Z = scale(PANSS.Pos, center = TRUE, scale = TRUE),
    TLI_DISORG_Z = scale(TLI_DISORG, center = TRUE, scale = TRUE)
  ) %>%
  # Calculate comScore by computing the row-wise mean of PANSS_Pos_Z and TLI_DISORG_Z
  mutate(
    comScore = rowMeans(select(., c("PANSS_Pos_Z", "TLI_DISORG_Z")), na.rm = TRUE)
  )

m_grand6 = lm(topic_mean ~ comScore, data = df4) 
summary(m_grand6)

