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
#prepare data: include all variables
#---------------------------------------------#
data <- read.csv(file = 'TOPSY_TwoGroups.csv')
df <- data[,c('ID','PatientCat','Gender','AgeScan1','SES','PANSS.Pos','Trails.B', 'Category.Fluency..animals.','DSST_Writen','DSST_Oral','TLI_DISORG','stim','n_1','n_2','n_3','n_4','n_5','num_all_words','num_content_words','num_repeated_words','n_sentence')]
df <- df[df$stim != 'Picture4', ]
df <- df[!is.na(df$n_1),]
df2 <- df %>%
  pivot_longer(cols = c(n_1, n_2, n_3, n_4, n_5), names_to = "wordpos", values_to = "w2v")

df3 <- df2 %>% select(ID, PatientCat, Gender, AgeScan1, TLI_DISORG, n_sentence, stim, wordpos, w2v) %>% 
  mutate(ID = as.factor(ID),
         PatientCat = as.factor(PatientCat),
         stim = as.factor(stim),
         wordpos = as.factor(wordpos)) %>%
         droplevels() %>%
         na.omit()

# Plot all data points
df3 %>% 
  ggplot(aes(x = PatientCat, y = w2v, col = PatientCat)) +
  geom_point() + 
  facet_grid(cols = vars(wordpos)) + 
  stat_summary(fun = mean, na.rm = TRUE, 
               geom = "point", shape = "diamond", 
               color = "black", size = 5)

# get mean and standard error values with standard error bars
summary_data <- df3 %>%
  group_by(wordpos, PatientCat) %>%
  summarise(mean_w2v = mean(w2v, na.rm = TRUE),
            sd_w2v = sd(w2v, na.rm = TRUE),
            se_w2v = sd(w2v, na.rm = TRUE) / sqrt(n()),
            n = n())

# Calculate 95% confidence interval
confidence_level <- 0.95
z_value <- qnorm((1 + confidence_level) / 2)
summary_data <- summary_data %>%
  mutate(ci_lower = mean_w2v - z_value * se_w2v,
         ci_upper = mean_w2v + z_value * se_w2v)

# Create the line plot with 95% confidence interval
ggplot(summary_data, aes(x = wordpos, y = mean_w2v, color = factor(PatientCat), group = PatientCat)) +
  geom_line() +
  geom_point(aes(shape = factor(PatientCat)), size = 3) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper, fill = factor(PatientCat)), alpha = 0.2, color = NA) +
  labs(x = "Word Position", y = "Mean w2v Value") +
  scale_color_manual(values = c("blue", "red"), labels = c("controls", "scz")) +
  scale_shape_manual(values = c(16, 17), labels = c("controls", "scz")) +
  scale_fill_manual(values = c("blue", "red"), labels = c("controls", "scz")) +
  theme_minimal()
ggsave("w2v_2groups.eps", device = "eps", width = 7, height = 5)

# Your existing code for creating the plot without semi-transparency
ggplot(summary_data, aes(x = wordpos, y = mean_w2v, color = factor(PatientCat), group = PatientCat)) +
  geom_line() +
  geom_point(aes(shape = factor(PatientCat)), size = 3) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper, fill = factor(PatientCat)), color = NA) +  # Removed alpha = 0.2
  labs(x = "Word Position", y = "Mean w2v Value") +
  scale_color_manual(values = c("blue", "red"), labels = c("controls", "scz")) +
  scale_shape_manual(values = c(16, 17), labels = c("controls", "scz")) +
  scale_fill_manual(values = c("blue", "red"), labels = c("controls", "scz")) +
  theme_minimal()

# Save the plot as an EPS file
ggsave("w2v_groups_noalpha.eps", device = "eps", width = 7, height = 5)


################Report this one################
#---------------------------------------------#
# run models: average across stim items
# individual reference level
# selecting different subsets of participants
#---------------------------------------------#
data <- read.csv(file = 'TOPSY_TwoGroups.csv')
df <- data[,c('ID','PatientCat','Gender','AgeScan1','SES','PANSS.Pos','Trails.B', 'Category.Fluency..animals.','DSST_Writen','DSST_Oral','TLI_DISORG','stim','n_1','n_2','n_3','n_4','n_5','num_all_words','num_content_words','num_repeated_words','n_sentence')]
df <- df[df$stim != 'Picture4', ]
df <- df[!is.na(df$n_1),]
df$DSST <- (df$DSST_Oral + df$DSST_Writen)/2

# calculate number of participants in each group
df9 <- df %>%
  group_by(ID) %>%
  summarise_all(mean, na.rm = TRUE)
sum(df9$PatientCat==1)
sum(df9$PatientCat==2)
df9 <- df9[!is.na(df9$TLI_DISORG),]

# convert the data to a long format
df2 <- df %>%
  pivot_longer(cols = c(n_1, n_2, n_3, n_4, n_5), names_to = "wordpos", values_to = "w2v")

# get mean values across stimuli
df3 <- df2 %>%
  group_by(ID, wordpos) %>%
  summarise(w2v_mean = mean(as.numeric(w2v), na.rm = TRUE),
            nsen_mean = mean(as.numeric(n_sentence), na.rm = TRUE),
            nword_mean = mean(as.numeric(num_all_words), na.rm = TRUE),
            ncontent_mean = mean(as.numeric(num_content_words), na.rm = TRUE),
            nrepeated_mean = mean(as.numeric(num_repeated_words), na.rm = TRUE)) %>%
  ungroup()

# Extracting other columns from the original dataframe for df3
other_columns <- df2 %>%
  select(ID, Gender, AgeScan1, SES, PANSS.Pos, PatientCat, TLI_DISORG, DSST, Trails.B, Category.Fluency..animals.) %>%
  distinct()

# Merging all the columns together based on 'ID'
df3 <- merge(df3, other_columns, by = "ID", all = TRUE)


#----------------------------
# get data containing all control demographic variables excluding SES: 34 HC + 70 FEP
df4 <- df3 %>% select(ID, PatientCat, Gender, AgeScan1, TLI_DISORG, nsen_mean, nword_mean, ncontent_mean, nrepeated_mean, wordpos, w2v_mean) %>% 
  mutate(ID = as.factor(ID),
         PatientCat = as.factor(PatientCat),
         Gender = as.factor(Gender),
         wordpos = as.factor(wordpos)) %>%
  droplevels() %>%
  na.omit()

# calculate number of participants in each group
df9 <- df4 %>%
  pivot_wider(names_from = wordpos, values_from = w2v_mean)
sum(df9$PatientCat==1) #HC: 34
sum(df9$PatientCat==2) #FEP: 70


#----------------------------
# get data containing all control demographic variables including SES: 33 HC + 60 FEP
df5 <- df3 %>% select(ID, PatientCat, Gender, AgeScan1, SES, TLI_DISORG, nsen_mean, nword_mean, ncontent_mean, nrepeated_mean, wordpos, w2v_mean) %>% 
  mutate(ID = as.factor(ID),
         PatientCat = as.factor(PatientCat),
         Gender = as.factor(Gender),
         wordpos = as.factor(wordpos)) %>%
  droplevels() %>%
  na.omit()

# calculate number of participants in each group
df9 <- df5 %>%
  pivot_wider(names_from = wordpos, values_from = w2v_mean)
sum(df9$PatientCat==1) #HC: 33
sum(df9$PatientCat==2) #FEP: 60

#----------------------------
# get data containing all control variables, SES + cognitive functions: 29 HC + 42 FEP
df6 <- df3 %>% select(ID, PatientCat, Gender, AgeScan1, SES, DSST, Trails.B, Category.Fluency..animals., TLI_DISORG, nsen_mean, nword_mean, ncontent_mean, nrepeated_mean, wordpos, w2v_mean) %>% 
  mutate(ID = as.factor(ID),
         PatientCat = as.factor(PatientCat),
         Gender = as.factor(Gender),
         wordpos = as.factor(wordpos)) %>%
  droplevels() %>%
  na.omit()

# calculate number of participants in each group
df9 <- df6 %>%
  pivot_wider(names_from = wordpos, values_from = w2v_mean)
sum(df9$PatientCat==1) #HC: 29
sum(df9$PatientCat==2) #FEP: 42


#----------------------------
# get data containing all variables, including SES, PANSS.Pos and cognitive functions: 24 HC + 40 FEP
df7 <- df3 %>% select(ID, PatientCat, Gender, AgeScan1, SES, PANSS.Pos, DSST, Trails.B, Category.Fluency..animals., TLI_DISORG, nsen_mean, nword_mean, ncontent_mean, nrepeated_mean, wordpos, w2v_mean) %>% 
  mutate(ID = as.factor(ID),
         PatientCat = as.factor(PatientCat),
         Gender = as.factor(Gender),
         wordpos = as.factor(wordpos)) %>%
  droplevels() %>%
  na.omit()

# calculate number of participants in each group
df9 <- df7 %>%
  pivot_wider(names_from = wordpos, values_from = w2v_mean)
sum(df9$PatientCat==1) #HC: 24
sum(df9$PatientCat==2) #FEP: 40


#------------------------------
# model continuous effect
#------------------------------
# all participants
m_grand4 = lmer(w2v_mean ~ wordpos*TLI_DISORG + (1 | ID) + ncontent_mean + nrepeated_mean + Gender + AgeScan1, data = df4) 
summary(m_grand4)

m_grand4b = lmer(w2v_mean ~ wordpos*TLI_DISORG + (1 | ID) + nsen_mean + Gender + AgeScan1, data = df4) 
summary(m_grand4b)

m_grand4c = lmer(w2v_mean ~ wordpos*TLI_DISORG + (1 | ID) + Gender + AgeScan1, data = df4) 
summary(m_grand4c)

# only participants with SES
m_grand5 = lmer(w2v_mean ~ wordpos*TLI_DISORG + (1 | ID) + Gender + AgeScan1 + SES + ncontent_mean + nrepeated_mean, data = df5) 
summary(m_grand5)

m_grand5b = lmer(w2v_mean ~ wordpos*TLI_DISORG + (1 | ID) + Gender + AgeScan1 + SES + nsen_mean, data = df5) 
summary(m_grand5b)

m_grand5c = lmer(w2v_mean ~ wordpos*TLI_DISORG + (1 | ID) + Gender + AgeScan1 + SES, data = df5) 
summary(m_grand5c)

# only participants with both SES and cognitive measures
m_grand6 = lmer(w2v_mean ~ wordpos*TLI_DISORG + (1 | ID) + SES + DSST + Trails.B + Category.Fluency..animals. + Gender + AgeScan1 + ncontent_mean + nrepeated_mean, data = df6) 
summary(m_grand6)

m_grand6b = lmer(w2v_mean ~ wordpos*TLI_DISORG + (1 | ID) + SES + DSST + Trails.B + Category.Fluency..animals. + Gender + AgeScan1 + nsen_mean, data = df6) 
summary(m_grand6b)

m_grand6c = lmer(w2v_mean ~ wordpos*TLI_DISORG + (1 | ID) + SES + DSST + Trails.B + Category.Fluency..animals. + Gender + AgeScan1, data = df6) 
summary(m_grand6c)

# only participants with SES, cognitive measures and PANSS.Pos
m_grand7 = lmer(w2v_mean ~ wordpos*PANSS.Pos + (1 | ID) + SES + Gender + AgeScan1 + ncontent_mean + nrepeated_mean, data = df7) 
summary(m_grand7)

m_grand7b = lmer(w2v_mean ~ wordpos*PANSS.Pos + (1 | ID) + SES + Gender + AgeScan1 + nsen_mean, data = df7) 
summary(m_grand7b)


#---------- group effect
# intercept being healthy controls, and n_1
# all participants
m_grand1 = lmer(w2v_mean ~ wordpos*PatientCat + (1 | ID) + ncontent_mean + nrepeated_mean + Gender + AgeScan1, data = df4) 
summary(m_grand1)

m_grand1b = lmer(w2v_mean ~ wordpos*PatientCat + (1 | ID) + nsen_mean + Gender + AgeScan1, data = df4) 
summary(m_grand1b)

# only participants with SES
m_grand2 = lmer(w2v_mean ~ wordpos*PatientCat + (1 | ID) + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals. + ncontent_mean + nrepeated_mean, data = df5) 
summary(m_grand2)

m_grand2b = lmer(w2v_mean ~ wordpos*PatientCat + (1 | ID) + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals. + nsen_mean, data = df5) 
summary(m_grand2b)

# only participants with both SES and cognitive measures
m_grand3 = lmer(w2v_mean ~ wordpos*PatientCat + (1 | ID) + SES + Gender + AgeScan1 + ncontent_mean + nrepeated_mean, data = df6) 
summary(m_grand3)

m_grand3b = lmer(w2v_mean ~ wordpos*PatientCat + (1 | ID) + SES + Gender + AgeScan1 + nsen_mean, data = df6) 
summary(m_grand3b)


# for each word position, check group effect
emm <- emmeans(m_grand1b, pairwise ~ PatientCat | wordpos, pbkrtest.limit = 14060)
pairs(emm, adjust = "tukey")


#---- combine variables
df8 <- df3 %>%
  mutate(
    PANSS_Pos_Z = scale(PANSS.Pos, center = TRUE, scale = TRUE),
    TLI_DISORG_Z = scale(TLI_DISORG, center = TRUE, scale = TRUE)
  ) %>%
  # Calculate comScore by computing the row-wise mean of PANSS_Pos_Z and TLI_DISORG_Z
  mutate(
    comScore = rowMeans(select(., c("PANSS_Pos_Z", "TLI_DISORG_Z")), na.rm = TRUE)
  )

m_grand8 = lmer(w2v_mean ~ wordpos*comScore + (1 | ID) + Gender + AgeScan1 + SES + ncontent_mean + nrepeated_mean, data = df8) 
summary(m_grand8)

m_grand8b = lmer(w2v_mean ~ wordpos*comScore  + (1 | ID) + Gender + AgeScan1 + SES + nsen_mean, data = df8) 
summary(m_grand8b)




#---------------------------------------------#
# run models: intercept being n_1
# take stim items as fixed effect
# every level compares to n_1
#---------------------------------------------#
cmat(design_matrix, add_intercept = TRUE) <- contrasts(df3$wordpos)
design_matrix
mpos1 <- lmer(w2v ~ PatientCat*wordpos + stim + Gender + AgeScan1 + SES + TLI_DISORG + n_sentence + (1 | ID), data = df3)
mpos2 <- lmer(w2v ~ PatientCat + wordpos + stim + Gender + AgeScan1 + SES + TLI_DISORG + n_sentence + (1 | ID), data = df3)
anova(mpos1,mpos2)
summary(mpos2)

df4 <- df3[!is.na(df3$TLI_DISORG), ]
m1 <- lmer(w2v ~ PatientCat*wordpos + stim + n_sentence + TLI_DISORG + n_sentence + (1 | ID), data = df4)
m2 <- lmer(w2v ~ PatientCat*wordpos + stim + n_sentence + (1 | ID), data = df4)
anova(m1, m2)

m3 <- lmer(w2v ~ PatientCat*wordpos + stim + (1 | ID), data = df3)
m4 <- lmer(w2v ~ PatientCat + wordpos + stim + (1 | ID), data = df3)
anova(m3,m4)

m5 <- lmer(w2v ~ PatientCat + wordpos + stim + (1 | ID), data = df3)
anova(m4,m5)

summary(m5)


#---------------------------------------------#
# run models: intercept being n_1
# take stim items as fixed effect
# all pairwise comparisons
#---------------------------------------------#
contrasts(df3$wordpos) <- contr.treatment(levels(df3$wordpos))
#contrasts(df3$PatientCat) <- contr.treatment(levels(df3$PatientCat))
contrasts(df3$PatientCat) <- c(-.5, .5)
cmat(design_matrix, add_intercept = TRUE) <- contrasts(df3$wordpos)
design_matrix

m5 <- lmer(w2v ~ PatientCat*wordpos + stim + Gender + AgeScan1 + SES + (1 + wordpos + stim| ID), data = df3, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
emm <- emmeans(m5, pairwise ~ wordpos | PatientCat, pbkrtest.limit = 14060)
pairs(emm, adjust = "tukey")
summary(m5)

#---------------------------------------------#
# run models: intercept being group average
# take stim items as fixed effect
# each wordpos level compares with n_1
#---------------------------------------------#
myMat <- solve(t(matrix(c(1/5,1/5,1/5,1/5,1/5,-1,1,0,0,0,-1,0,1,0,0,-1,0,0,1,0,-1,0,0,0,1), nrow =5, ncol = 5))) ## matrix for word_type
design_matrix <- hypr()
cmat(design_matrix, add_intercept = FALSE) <- myMat
design_matrix

contrasts(df3$wordpos) <- myMat
contrasts(df3$PatientCat) <- c(-.5, .5)

m5 = lmer(w2v ~ PatientCat + wordpos + stim + (1 | ID), data = df3)
summary(m5)

#---------------------------------------------#
# run models: intercept being group average
# take stim items as fixed effect
# step-wide pairwise comparisons
#---------------------------------------------#
contrasts(df3$wordpos) <- contr.sdif(5, contrasts = TRUE, sparse = FALSE)
cmat(design_matrix, add_intercept = TRUE) <- contrasts(df3$wordpos)
design_matrix

contrasts(df3$PatientCat) <- c(-.5, .5)

contrasts(df3$stim) <- contr.sdif(4, contrasts = TRUE, sparse = FALSE)

#run model
m4 = lmer(w2v ~ PatientCat*wordpos + stim + (1 | ID), data = df3)
m5 = lmer(w2v ~ PatientCat + wordpos + stim + (1 | ID), data = df3)
m6 = lmer(w2v ~ PatientCat + wordpos + stim + (1 + wordpos|| ID), data = df3)
anova(m4,m5)
anova(m6,m5)
summary(m5)
summary(m4)



#---------------------------------------------#
# run models: average across stim items
# intercept being group average
# step-wide pairwise comparisons
#---------------------------------------------#

contrasts(df4$wordpos) <- contr.sdif(5, contrasts = TRUE, sparse = FALSE)
contrasts(df4$PatientCat) <- c(-.5, .5)


#---------------------------------------------#
# run models: average across stim items
# intercept being wordpos average
#---------------------------------------------#
myMat <- solve(t(matrix(c(1/5,1/5,1/5,1/5,1/5,-1,1,0,0,0,-1,0,1,0,0,-1,0,0,1,0,-1,0,0,0,1), nrow =5, ncol = 5))) ## matrix for word_type
design_matrix <- hypr()
cmat(design_matrix, add_intercept = FALSE) <- myMat
design_matrix

contrasts(df4$wordpos) <- myMat
contrasts(df4$PatientCat) <- c(-.5, .5)

m5 = lmer(w2v_mean ~ wordpos*TLI_DISORG + (1 | ID) + nsen_mean, data = df4)
summary(m5)
anova(m5)


