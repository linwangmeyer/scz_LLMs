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
p_load(interactions,lavaan,psych, readxl, semPlot)

#setwd(dirname(getActiveDocumentContext()$path))    ## sets dir to R script path
setwd("/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/")


#---------------------------------------------#
#prepare data
#---------------------------------------------#
data <- read.csv(file = 'TOPSY_TwoGroups_1min.csv')
df <- data[,c('ID','PatientCat','Gender','AgeScan1','SES','PANSS.Pos','stim','TLI_DISORG','num_all_words','num_content_words','num_repetition','nsen','entropyApproximate')]
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
#vioplot(x1, x2, names=c("ctrl", "scz"), colors=c("blue", "red"))


############################run models###################
#-----------------------------------------------------------#
# run models: average across stim items --> linear regression model
# individual reference level; include all predictors
# different subsets of participants, depending on the included variables
#-----------------------------------------------------------#

#-------------------------------
# get data ready
data <- read.csv(file = 'TOPSY_TwoGroups_1min.csv')
#data <- read.csv(file = 'TOPSY_TwoGroups_spontaneous.csv')
#data <- read.csv(file = 'TOPSY_TwoGroups_concatenated.csv')

df <- data[,c('ID','PatientCat','Gender','AgeScan1','SES','PANSS.Pos','Trails.B', 'Category.Fluency..animals.','DSST_Writen','DSST_Oral','TLI_DISORG','TLI_IMPOV','nsen', 'num_all_words', 'num_content_words','num_repetition','stim','entropyApproximate','entropyTransform','TransformSimilarity')]
df <- df[df$stim != 'Picture4', ]
df <- df[!is.na(df$entropyApproximate),]

df$DSST <- (df$DSST_Oral + df$DSST_Writen)/2

#average across stims
df2 <- df %>%
  group_by(ID) %>%
  summarise(topic_mean = mean(as.numeric(entropyApproximate), na.rm = TRUE),
            topicSim_mean = mean(as.numeric(TransformSimilarity), na.rm = TRUE),
            nword_mean = mean(as.numeric(num_all_words), na.rm = TRUE),
            ncontent_mean = mean(as.numeric(num_content_words), na.rm = TRUE),
            nrepeated_mean = mean(as.numeric(num_repetition), na.rm = TRUE),
            nsen_mean = mean(as.numeric(nsen), na.rm = TRUE)) %>%
  ungroup()

# Extracting other columns from the original dataframe for df3
other_columns <- df %>%
  select(ID, PatientCat, Gender, AgeScan1, SES, PANSS.Pos, TLI_DISORG, TLI_IMPOV, DSST, Trails.B, Category.Fluency..animals.) %>%
  distinct()

# Merging all the columns together based on 'ID'
df3 <- merge(df2, other_columns, by = "ID", all = TRUE)

#----------------------------
# get data containing all control demographic variables excluding SES: 34 HC + 70 FEP
df4 <- df3 %>%
  select(ID, PatientCat, Gender, AgeScan1, TLI_DISORG, TLI_IMPOV, nsen_mean, nword_mean, ncontent_mean, nrepeated_mean, topic_mean) %>% 
  mutate(across(c(ID, PatientCat, Gender), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values

sum(df4$PatientCat==1) #HC: 34
sum(df4$PatientCat==2) #FEP: 70

#summary(df4$nword_mean)
#df4 <- df4[df4$nword_mean>90,]


#df4 <- df4[df4$PatientCat==2,]

#----------------------------
# get data containing all control demographic variables including SES: 33 HC + 60 FEP
df5 <- df3 %>%
  select(ID, PatientCat, Gender, AgeScan1, SES, TLI_DISORG, TLI_IMPOV, nsen_mean, nword_mean, ncontent_mean, nrepeated_mean, topic_mean) %>% 
  mutate(across(c(ID, PatientCat, Gender), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values

sum(df5$PatientCat==1) #HC: 33
sum(df5$PatientCat==2) #FEP: 60

#----------------------------
# get data containing all control variables, SES + cognitive functions: 29 HC + 42 FEP
df6 <- df3 %>%
  select(ID, PatientCat, Gender, AgeScan1, SES, DSST, Trails.B, Category.Fluency..animals., TLI_DISORG, TLI_IMPOV, nsen_mean, nword_mean, ncontent_mean, nrepeated_mean, topic_mean) %>% 
  mutate(across(c(ID, PatientCat, Gender), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values

sum(df6$PatientCat==1) #HC: 29
sum(df6$PatientCat==2) #FEP: 42

#----------------------------
# get data containing all variables, including SES, PANSS.Pos and cognitive functions: 24 HC + 40 FEP
df7 <- df3 %>%
  select(ID, PatientCat, Gender, AgeScan1, SES, PANSS.Pos, DSST, Trails.B, Category.Fluency..animals., TLI_DISORG, TLI_IMPOV, nsen_mean, nword_mean, ncontent_mean, nrepeated_mean, topic_mean) %>% 
  mutate(across(c(ID, PatientCat, Gender), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values

sum(df7$PatientCat==1) #HC: 24
sum(df7$PatientCat==2) #FEP: 40

#----------------------------
# scatter plot
df_plot <- aggregate(cbind(TLI_DISORG, nword_mean, topic_mean) ~ ID + PatientCat, data=df4, FUN=mean)
labels <- c("Controls", "FEP")

# Scatter plot of TLI_DISORG vs. entropyApproximate
p1 <- ggplot(df_plot, aes(x=TLI_DISORG, y=topic_mean, color=factor(PatientCat))) +
  geom_point() +
  labs(x="TLI_DISORG", y="Entropy Approximate") +
  scale_color_manual(values=c("blue", "red"), labels = labels) +
  ggtitle("TLI_DISORG vs. Entropy") +
  theme_minimal()
ggsave("../plots/TLI_entropy.png", plot = p1, width = 6, height = 4, units = "in", dpi = 300)

# Scatter plot of TLI_DISORG vs. nword
p2 <- ggplot(df_plot, aes(x=TLI_DISORG, y=nword_mean, color=factor(PatientCat))) +
  geom_point() +
  labs(x="TLI_DISORG", y="nword") +
  scale_color_manual(values=c("blue", "red"), labels = labels) +
  ggtitle("TLI_DISORG vs. nword") +
  theme_minimal()
ggsave("../plots/TLI_nword.png", plot = p2, width = 6, height = 4, units = "in", dpi = 300)

# Scatter plot of nword vs. entropyApproximate
p3 <- ggplot(df_plot, aes(x=nword_mean, y=topic_mean, color=factor(PatientCat))) +
  geom_point() +
  labs(x="nword", y="Entropy Approximate") +
  scale_color_manual(values=c("blue", "red"), labels = labels) +
  ggtitle("nword vs. Entropy") +
  theme_minimal()
ggsave("../plots/nword_entropy.png", plot = p3, width = 6, height = 4, units = "in", dpi = 300)



#----------------------------------------------------------
# mean center data to get effect at the mean centered level
d1 <- df4 %>%
  mutate(
  nword_centered = scale(nword_mean, scale=FALSE),
  ncontent_centered = scale(ncontent_mean, scale = FALSE),
  nrepeated_centered = scale(nrepeated_mean, scale = FALSE),
  TLI_pos_centered = scale(TLI_DISORG, scale = FALSE),
  TLI_neg_centered = scale(TLI_IMPOV, scale = FALSE)
)

d2 <- df5 %>%
  mutate(
    nword_centered = scale(nword_mean, scale=FALSE),
    ncontent_centered = scale(ncontent_mean, scale = FALSE),
    nrepeated_centered = scale(nrepeated_mean, scale = FALSE),
    TLI_pos_centered = scale(TLI_DISORG, scale = FALSE),
    TLI_neg_centered = scale(TLI_IMPOV, scale = FALSE),
    SES_centered = scale(SES, scale = FALSE)
    )

d3 <- df6 %>%
  mutate(
    nword_centered = scale(nword_mean, scale=FALSE),
    ncontent_centered = scale(ncontent_mean, scale = FALSE),
    nrepeated_centered = scale(nrepeated_mean, scale = FALSE),
    TLI_pos_centered = scale(TLI_DISORG, scale = FALSE),
    TLI_neg_centered = scale(TLI_IMPOV, scale = FALSE),
    WM_centered = scale(DSST, scale = FALSE),
    ExControl_centered = scale(Trails.B, scale = FALSE),
    SemFluency_centered = scale(Category.Fluency..animals., scale = FALSE)
  )

contrasts(d1$Gender) <- c(-.5, .5)
contrasts(d1$PatientCat) <- c(-.5, .5)

contrasts(d2$Gender) <- c(-.5, .5)
contrasts(d2$PatientCat) <- c(-.5, .5)

contrasts(d3$Gender) <- c(-.5, .5)
contrasts(d3$PatientCat) <- c(-.5, .5)

#--------------------------------------------------------
# test interaction between TLI_pos and nword
m1 = lm(topic_mean ~ TLI_neg_centered*nword_centered + Gender + AgeScan1, data = d1) 
summary(m1)

m2 = lm(topic_mean ~ TLI_pos_centered*nword_centered + SES + Gender + AgeScan1, data = d2) 
summary(m2)

m3 = lm(topic_mean ~ TLI_pos_centered*nword_centered + SES + WM_centered + ExControl_centered + SemFluency_centered + Gender + AgeScan1, data = d3) 
summary(m3)

#-------------------------------
## Calculate the trends by condition: for each level of nword_centered
emTrends_m1 <- emtrends(m1, "nword_centered", var = "TLI_pos_centered",
                        at=list(nword_centered = c(min(d1$nword_centered), #n=-78
                                                quantile(d1$nword_centered, probs = 0.25), # 1st quantile: n=-24
                                                0, # mean: n=152.10
                                                quantile(d1$nword_centered, probs = 0.5), # median: n=-0.8 
                                                quantile(d1$nword_centered, probs = 0.75), # 3rd quantile: n=22
                                                 max(d1$nword_centered)))) # max: n=110
summary(emTrends_m1, infer= TRUE)

# visualize the interactions: TLI vs. entropy for each Nword level
m1 %>%
  interactions::interact_plot(pred = TLI_pos_centered,
                              modx = nword_centered,
                              modx.values = c(-50,0,50,100),
                              interval = TRUE,
                              int.type = "confidence",
                              legend.main = "Nword_meanCentered:") +
  labs(x = "TLI",
       y = "Entropy") +
  geom_hline(yintercept = 0) +
  theme_bw() + ylim(8,12) +
  theme(#legend.position = c(0, 1),
    #legend.justification = c(-0.1, 1.1),
    legend.background = element_rect(color = "black"),
    legend.key.width = unit(1.5, "cm"))


# Create bins for TLI_disorg and nword_mean
d1$TLI_bin <- cut(d1$TLI_pos_centered,
                         breaks = c(-Inf, 1, 3, Inf),
                         labels = c('low','medium','high'),
                         right = FALSE)

d1$nword_bin <- cut(d1$nword_centered, 
                         breaks = c(-Inf, -50, 0, 50, Inf), 
                         labels = c("short", 'short_medium', "long_medium", "long"),
                         right = FALSE)

# Create the bar plot
ggplot(d1, aes(x = TLI_bin, y = topic_mean, fill = nword_bin)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "TLI_pos", y = "Entropy") +
  scale_fill_discrete(name = "nword") +  # Customize legend title
  theme_minimal()



## Calculate the trends by condition: for each level of TLI
emTrends_m1 <- emtrends(m1, "TLI_pos_centered", var = "nword_centered",
                        at=list(TLI_pos_centered = c(-0.5,0,0.5,1,5))) #max
summary(emTrends_m1, infer= TRUE)

# visualize the interactions: TLI vs. nwords for each TLI level
m1 %>%
  interactions::interact_plot(pred = nword_centered,
                              modx = TLI_pos_centered,
                              modx.values = c(-0.7259615,0,0.3365,3),
                              interval = TRUE,
                              int.type = "confidence",
                              legend.main = "TLI:") +
  labs(x = "Nwords",
       y = "Entropy") +
  geom_hline(yintercept = 0) +
  theme_bw() + ylim(8,12) +
  theme(#legend.position = c(0, 1),
    #legend.justification = c(-0.1, 1.1),
    legend.background = element_rect(color = "black"),
    legend.key.width = unit(1.5, "cm"))



#---------- group categorical effect
# all participants
d1$PatientCat <- relevel(factor(d1$PatientCat), ref = '1')
contrasts(d1$PatientCat) <- c(-.5, .5)
m4 = lm(topic_mean ~ PatientCat*nword_centered + Gender + AgeScan1, data = d1) 
summary(m4)
#anova(m4)

# only participants with SES
m5 = lm(topic_mean ~ PatientCat*nword_centered + Gender + AgeScan1 + SES, data = d2) 
summary(m5)


# only participants with both SES and cognitive measures
m6 = lm(topic_mean ~ PatientCat*nword_centered + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals., data = d3) 
summary(m6)




#--------------------------------------------------------
# mediation analysis
d1 <- df4 %>%
  mutate(
    nword_centered = scale(nword_mean, scale=TRUE),
    TLI_centered = scale(TLI_IMPOV, scale = TRUE)
  )

# check correlation
correlations <- cor(d1 %>% select(TLI_centered,nword_centered,topic_mean))
print(correlations)

mediation_model <- '
# direct effect
nword_centered ~ a * TLI_centered
topic_mean ~ c * TLI_centered + b * nword_centered

# indirect effect (a*b)
indirect := a * b

#Total effect (c+indirect)
total := c + indirect
'

# Estimate the mediation model
mediation_results <- sem(mediation_model, data = d1)

# Summrize the results
summary(mediation_results, standardized =TRUE, fit.measures = TRUE)

# visualize
semPaths(mediation_results, whatLabels = 'est',
         style = 'lisrel',
         intercepts=FALSE)

#--------------------------------------------------------
# lmer models: without interactions
# all participants
m_grand4 = lm(topic_mean ~ TLI_DISORG + nword_mean + Gender + AgeScan1, data = df4) 
summary(m_grand4)

# check multicollinearity
car::vif(m_grand4)


m_grand4b = lm(topic_mean ~ TLI_DISORG + Gender + AgeScan1, data = df4) 
summary(m_grand4b)

m_grand4c = lm(topic_mean ~ TLI_IMPOV + Gender + AgeScan1, data = df4) 
summary(m_grand4c)

m_grand4d = lm(topic_mean ~ TLI_IMPOV + ncontent_mean + Gender + AgeScan1, data = df4) 
summary(m_grand4d)

# only participants with SES
m_grand5 = lm(topic_mean ~ TLI_IMPOV + nword_mean + Gender + AgeScan1 + SES, data = df5) 
summary(m_grand5)

m_grand5b = lm(topic_mean ~ TLI_IMPOV + nsen_mean + Gender + AgeScan1 + SES, data = df5) 
summary(m_grand5b)

m_grand5c = lm(topic_mean ~ TLI_IMPOV + Gender + AgeScan1 + SES, data = df5) 
summary(m_grand5c)

m_grand5d = lm(topic_mean ~ TLI_IMPOV + ncontent_mean + Gender + AgeScan1 + SES, data = df5) 
summary(m_grand5d)

# only participants with both SES and cognitive measures
m_grand6 = lm(topic_mean ~ TLI_IMPOV + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals. + nword_mean, data = df6) 
summary(m_grand6)

m_grand6b = lm(topic_mean ~ TLI_IMPOV + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals. + nsen_mean, data = df6) 
summary(m_grand6b)

m_grand6c = lm(topic_mean ~ TLI_IMPOV + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals., data = df6) 
summary(m_grand6c)

m_grand6d = lm(topic_mean ~ TLI_IMPOV + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals. + ncontent_mean, data = df6) 
summary(m_grand6d)

# only participants with SES, cognitive measures and PANSS.Pos: 24 HC + 40 FEP
m_grand7 = lm(topic_mean ~ TLI_IMPOV + PANSS.Pos + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals. + nword_mean, data = df7) 
summary(m_grand7)

m_grand7b = lm(topic_mean ~ TLI_IMPOV + PANSS.Pos + Gender + AgeScan1 + SES + DSST + Trails.B + Category.Fluency..animals. + nsen_mean, data = df7) 
summary(m_grand7b)


#---------- continuous effect
# only patients
df8 <- df3 %>%
  filter(PatientCat == '2')

m_grand4 = lm(topic_mean ~ TLI_DISORG + nword_mean + Gender + AgeScan1, data = df4) 
summary(m_grand4)

m_grand4b = lm(topic_mean ~ TLI_DISORG + nsen_mean + Gender + AgeScan1, data = df4) 
summary(m_grand4b)

m_grand4c = lm(topic_mean ~ TLI_DISORG + Gender + AgeScan1, data = df4) 
summary(m_grand4c)

m_grand4d = lm(topic_mean ~ TLI_DISORG + ncontent_mean + Gender + AgeScan1, data = df4) 
summary(m_grand4d)




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