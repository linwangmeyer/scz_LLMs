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

df <- data[,c('ID','PatientCat','Gender','AgeScan1','TLI_DISORG','n_sentence','stim','n_1','n_2','n_3','n_4','n_5')]
df2 <- df %>%
  pivot_longer(cols = c(n_1, n_2, n_3, n_4, n_5), names_to = "wordpos", values_to = "w2v")

df3 <- df2 %>% select(ID, PatientCat, TLI_DISORG, n_sentence, stim, wordpos, w2v) %>% 
  mutate(ID = as.factor(ID),
         PatientCat = as.factor(PatientCat),
         stim = as.factor(stim),
         wordpos = as.factor(wordpos)) %>%
         droplevels() %>%
         na.omit()

# Plot
df3 %>% 
  ggplot(aes(x = PatientCat, y = w2v, col = PatientCat)) +
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


#---------------------------------------------#
# run models: intercept being n_1
# take stim items as fixed effect
# every level compares to n_1
#---------------------------------------------#
cmat(design_matrix, add_intercept = TRUE) <- contrasts(df3$wordpos)
design_matrix
mpos1 <- lmer(w2v ~ PatientCat*wordpos + stim + n_sentence + TLI_DISORG + n_sentence + (1 | ID), data = df3)
anova(mpos1)
summary(mpos1)

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

m5 <- lmer(w2v ~ PatientCat*wordpos + stim + (1 + wordpos + stim| ID), data = df3, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
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
################Report this one################
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
df5 <- df3 %>% ungroup() %>% 
  group_by(PatientCat, wordpos, ID) %>% 
  summarize(grandW2V = mean(w2v, na.rm = TRUE))

contrasts(df5$wordpos) <- contr.sdif(5, contrasts = TRUE, sparse = FALSE)
contrasts(df5$PatientCat) <- c(-.5, .5)


m_grand1 = lmer(grandW2V ~ PatientCat*wordpos + (1 | ID), data = df5) 
m_grand2 = lmer(grandW2V ~ PatientCat + wordpos + (1 | ID), data = df5) 
anova(m_grand1,m_grand2)

m_grand3 = lmer(grandW2V ~ PatientCat*wordpos + (1 | ID), data = df5) 
summary(m_grand3)

emm <- emmeans(m_grand3, pairwise ~ PatientCat | wordpos, pbkrtest.limit = 14060)
pairs(emm, adjust = "tukey")

