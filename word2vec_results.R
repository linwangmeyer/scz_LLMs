rm(list=ls())

# -- Set function defaults:
filter <- dplyr::filter
group_by <- dplyr::group_by
summarize <- dplyr::summarize
select <- dplyr::select
rename <- dplyr::rename

# -- Set working directory and install packages:
if(!require(pacman)) {install.packages("pacman"); require(pacman)}

p_load("ggplot2", "rstudioapi", "tidyverse", "lme4", "lmerTest", 
       "car", "patchwork", "afex", "yarrr", "hypr", "MASS", 
       "emmeans", "udpipe")

#setwd(dirname(getActiveDocumentContext()$path))    ## sets dir to R script path
setwd("/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/")

#---------------------------------------------
# prepare data: include all variables
#---------------------------------------------
data <- read.csv(file = 'TOPSY_TwoGroups.csv')
df <- data[,c('ID','PatientCat','Gender','AgeScan1','SES','PANSS.Pos','TLI_DISORG','n_sentence','stim','n_1','n_2','n_3','n_4','n_5')]
df <- df[df$stim != 'Picture4', ]
df <- df[!is.na(df$n_1),]
df2 <- df %>%
  pivot_longer(cols = c(n_1, n_2, n_3, n_4, n_5), names_to = "wordpos", values_to = "w2v")

df3 <- df2 %>% select(ID, PatientCat, Gender, AgeScan1, SES, TLI_DISORG, n_sentence, stim, wordpos, w2v) %>% 
  mutate(ID = as.factor(ID),
         PatientCat = as.factor(PatientCat),
         stim = as.factor(stim),
         wordpos = as.factor(wordpos)) %>%
  droplevels() %>%
  na.omit()

# Plot all data points
df3 %>% 
  ggplot(aes(x = PatientCat, y = mean_w2v, col = PatientCat)) +
  geom_point() + 
  facet_grid(cols = vars(wordpos)) + 
  stat_summary(fun = mean, na.rm = TRUE, 
               geom = "point", shape = "diamond", 
               color = "black", size = 5)

# get mean and standard error values
summary_data <- df3 %>%
  group_by(wordpos, PatientCat) %>%
  summarise(mean_w2v = mean(w2v, na.rm = TRUE),
            se_w2v = sd(w2v, na.rm = TRUE) / sqrt(n()))

# Create the line plot with standard error bars
summary_data <- df3 %>%
  group_by(wordpos, PatientCat) %>%
  summarise(mean_w2v = mean(w2v, na.rm = TRUE),
            sd_w2v = sd(w2v, na.rm = TRUE),
            n = n())

# Calculate standard error
summary_data <- summary_data %>%
  mutate(se_w2v = sd_w2v / sqrt(n))

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


################Report this one################
#---------------------------------------------
# run models: average across stim items
# individual reference level
# include all control predictors
#---------------------------------------------
data <- read.csv(file = 'TOPSY_TwoGroups.csv')
df <- data[,c('ID','PatientCat','Gender','AgeScan1','SES','PANSS.Pos','TLI_DISORG','n_sentence','stim','n_1','n_2','n_3','n_4','n_5')]
df <- df[df$stim != 'Picture4', ]
df <- df[!is.na(df$n_1),]
df2 <- df %>%
  pivot_longer(cols = c(n_1, n_2, n_3, n_4, n_5), names_to = "wordpos", values_to = "w2v")

mean_w2v <- df2 %>%
  group_by(wordpos, ID) %>%
  summarize(mean_w2v = mean(w2v, na.rm = TRUE))

df3 <- df2 %>%
  select(-stim) %>%
  distinct() %>%
  left_join(mean_w2v, by = c("wordpos", "ID"))

df4 <- df3 %>% select(ID, PatientCat, Gender, AgeScan1, SES, PANSS.Pos, TLI_DISORG, n_sentence, wordpos, mean_w2v) %>% 
  mutate(ID = as.factor(ID),
         PatientCat = as.factor(PatientCat),
         Gender = as.factor(Gender),
         wordpos = as.factor(wordpos)) %>%
  droplevels() %>%
  na.omit()


#---------- group effect
# intercept being healthy controls, and n_1
m_grand1 = lmer(mean_w2v ~ wordpos + (1 | ID) + Gender + AgeScan1 + SES, data = df4) 
m_grand2 = lmer(mean_w2v ~ wordpos + PatientCat + (1 | ID) + Gender + AgeScan1 + SES, data = df4) 
anova(m_grand1,m_grand2)

m_grand3 = lmer(mean_w2v ~ wordpos*PatientCat + (1 | ID) + Gender + AgeScan1 + SES, data = df4) 
anova(m_grand3,m_grand2)
summary(m_grand2)
summary(m_grand3)

# for each word position, group effect
emm <- emmeans(m_grand3, pairwise ~ PatientCat | wordpos, pbkrtest.limit = 14060)
pairs(emm, adjust = "tukey")

# for each group, the word position effect
emm <- emmeans(m_grand3, pairwise ~ wordpos | PatientCat, pbkrtest.limit = 14060)
pairs(emm, adjust = "tukey")

#----------
#continuous effect
m_grand4 = lmer(mean_w2v ~ wordpos + (1 | ID) + Gender + AgeScan1 + SES + n_sentence + PANSS.Pos, data = df4) 
summary(m_grand4)

m_grand5 = lmer(mean_w2v ~ wordpos + (1 | ID) + Gender + AgeScan1 + SES + n_sentence + TLI_DISORG, data = df4) 
summary(m_grand5)

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

m_grand6 = lmer(mean_w2v ~ wordpos + (1 | ID) + Gender + AgeScan1 + SES + n_sentence + comScore, data = df4) 
summary(m_grand6)





#---------------------------------------------
# run models: average across stim items
# individual reference level
# include only critical predictors (more data)
#---------------------------------------------
data <- read.csv(file = 'TOPSY_TwoGroups.csv')
df <- data[,c('ID','PatientCat','PANSS.Pos','TLI_DISORG','n_sentence','stim','n_1','n_2','n_3','n_4','n_5')]
df <- df[df$stim != 'Picture4', ]
df <- df[!is.na(df$n_1),]
df2 <- df %>%
  pivot_longer(cols = c(n_1, n_2, n_3, n_4, n_5), names_to = "wordpos", values_to = "w2v")

mean_w2v <- df2 %>%
  group_by(wordpos, ID) %>%
  summarize(mean_w2v = mean(w2v, na.rm = TRUE))

df3 <- df2 %>%
  select(-stim) %>%
  distinct() %>%
  left_join(mean_w2v, by = c("wordpos", "ID"))

df4 <- df3 %>% select(ID, PatientCat, PANSS.Pos, TLI_DISORG, n_sentence, wordpos, mean_w2v) %>% 
  mutate(ID = as.factor(ID),
         PatientCat = as.factor(PatientCat),
         wordpos = as.factor(wordpos)) %>%
  droplevels() %>%
  na.omit()


#----------
# intercept being healthy controls and n_1
m_grand1 = lmer(mean_w2v ~ wordpos + (1 | ID), data = df4) 
m_grand2 = lmer(mean_w2v ~ wordpos + PatientCat + (1 | ID), data = df4) 
anova(m_grand1,m_grand2)

m_grand3 = lmer(mean_w2v ~ wordpos*PatientCat + (1 | ID), data = df4) 
anova(m_grand3,m_grand2)

#for PatientCat=1 healthy group:
summary(m_grand2)
summary(m_grand3)

# for each word position, group effect
emm <- emmeans(m_grand3, pairwise ~ PatientCat | wordpos, pbkrtest.limit = 14060)
pairs(emm, adjust = "tukey")

# for each group, the word position effect
emm <- emmeans(m_grand3, pairwise ~ wordpos | PatientCat, pbkrtest.limit = 14060)
pairs(emm, adjust = "tukey")

#----------
#continuous effect
m_grand4 = lmer(mean_w2v ~ wordpos + (1 | ID) + Gender + AgeScan1 + SES + n_sentence + PANSS.Pos, data = df4) 
summary(m_grand4)

m_grand5 = lmer(mean_w2v ~ wordpos + (1 | ID) + Gender + AgeScan1 + SES + n_sentence + TLI_DISORG, data = df4) 
summary(m_grand5)

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

m_grand6 = lmer(mean_w2v ~ wordpos + (1 | ID) + n_sentence + comScore, data = df4) 
summary(m_grand6)