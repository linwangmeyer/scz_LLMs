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
       "emmeans", "udpipe",'glmnet')
p_load(interactions,lavaan,psych, readxl, semPlot, MASS, car,glmnet)

current_path <- dirname(getActiveDocumentContext()$path)
subdirectory <- "stimuli/Relabeld/analysis"
new_path <- file.path(dirname(current_path), subdirectory)
setwd(new_path)

fname = file.path(new_path,'/TOPSY_all_spontaneous.csv')
data = read.csv(fname)

############################run models###################
#-----------------------------------------------------------#
# run models: average across stim items --> linear regression model
# select important features
# different subsets of participants, depending on the included variables
#-----------------------------------------------------------#

#-------------------------------
# get data ready
df <- data[,c('ID', 'Age', 'PatientCat', 'Gender', 'SES', 'PANSS_Total', 'PANSS_Neg',
              'PANSS_Pos', 'PANSS_p2', 'Trails_B', 'DSST_Writen', 'DSST_Oral',
              'DSST_Total', 'SemFluency', 'DUP_weeks', 'SOFAS', 'TLI_IMPOV',
              'TLI_DISORG', 'TLI_Total', 'entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4',
              'n_5', 'num_all_words', 'num_content_words', 'num_repetition',
              'consec_mean', 's0_mean', 'n_segment', 'senN_4', 'senN_3', 'senN_2', 'senN_1',
              'N_fillers', 'false_starts','self_corrections', 'clause_density',
              'dependency_distance', 'content_function_ratio', 'type_token_ratio',
              'average_word_frequency')]
df <- df[!is.na(df$n_1),]
df$w2v <- (df$n_1 + df$n_2 + df$n_3 + df$n_4 + df$n_5)/5
df$sensim <- (df$senN_1 + df$senN_2 + df$senN_3 + df$senN_4)/4

#average across stims
df2 <- df %>%
  group_by(ID) %>%
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE))) %>%
  ungroup() %>%
  filter(PatientCat %in% c(1, 2))


#----------------------------
# get data containing all control demographic variables excluding SES: 34 HC + 70 FEP
df4 <- df2 %>%
  select(ID, PatientCat, Gender, Age, TLI_DISORG, TLI_IMPOV, 
         n_segment, num_all_words, num_content_words, num_repetition,
         N_fillers,content_function_ratio,type_token_ratio,average_word_frequency,
         entropyApproximate, clause_density, dependency_distance,
         s0_mean,w2v) %>% 
  mutate(across(c(ID, PatientCat, Gender), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values

sum(df4$PatientCat==1) #HC: 34
sum(df4$PatientCat==2) #FEP: 70


#----------------------------
# get data containing all control variables, SES: 29 HC + 42 FEP
df5 <- df2 %>%
  select(ID, PatientCat, Gender, Age, TLI_DISORG, TLI_IMPOV, 
         n_segment, num_all_words, num_content_words, num_repetition,
         N_fillers,content_function_ratio,type_token_ratio,average_word_frequency,
         entropyApproximate, clause_density, dependency_distance,
         s0_mean,w2v,SES) %>% 
  mutate(across(c(ID, PatientCat, Gender), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values

sum(df5$PatientCat==1) #HC: 33
sum(df5$PatientCat==2) #FEP: 60

#----------------------------
# get data containing all control variables, SES + cognitive functions: 29 HC + 42 FEP
df6 <- df2 %>%
  select(ID, PatientCat, Gender, Age, TLI_DISORG, TLI_IMPOV, 
         n_segment, num_all_words, num_content_words, num_repetition,
         N_fillers,content_function_ratio,type_token_ratio,average_word_frequency,
         entropyApproximate, clause_density, dependency_distance,
         s0_mean,w2v,SES, Trails_B, DSST_Writen, DSST_Oral,
         DSST_Total, SemFluency) %>% 
  mutate(across(c(ID, PatientCat, Gender), as.factor)) %>%
  drop_na()#na.omit()# Remove rows with NA values

sum(df6$PatientCat==1) #HC: 29
sum(df6$PatientCat==2) #FEP: 42

# mean center data
continuous_vars <- df4 %>%
  select(-c(ID, PatientCat, Gender)) %>%
  names()

d1 <- df4 %>%
  mutate(across(all_of(continuous_vars), ~ scale(.x, scale = FALSE), .names = "{.col}_centered"))
d2 <- df5 %>%
  mutate(across(all_of(continuous_vars), ~ scale(.x, scale = FALSE), .names = "{.col}_centered"))
d3 <- df6 %>%
  mutate(across(all_of(continuous_vars), ~ scale(.x, scale = FALSE), .names = "{.col}_centered"))

# recode categorical data
contrasts(d1$Gender) <- c(-.5, .5)
contrasts(d1$PatientCat) <- c(-.5, .5)
contrasts(d2$Gender) <- c(-.5, .5)
contrasts(d2$PatientCat) <- c(-.5, .5)
contrasts(d3$Gender) <- c(-.5, .5)
contrasts(d3$PatientCat) <- c(-.5, .5)


#--------------------------------------------------------
# test effects
m1 = lm(clause_density ~ TLI_IMPOV_centered + Gender + Age_centered, data = d1) 
summary(m1)
m2 = lm(clause_density ~ TLI_IMPOV_centered + Gender, data = d1) 
summary(m2)
m3 = lm(clause_density ~ TLI_IMPOV_centered, data = d1) 
summary(m3)
anova(m1,m3)



# --------
mean_centered_vars <- d1 %>%
  select(ends_with("_centered")) %>%
  select(-c(TLI_DISORG_centered, TLI_IMPOV_centered,
            nword_centered, ncontent_centered,
            nrepeated_centered)) %>%
  names()

# Fit model to predict negative symptoms
formula <- as.formula(paste("TLI_IMPOV_centered ~", paste(mean_centered_vars, collapse = " + ")))
full_model <- lm(formula, data = d1)

step_model <- step(full_model, direction = "backward")
summary(step_model)

# Check for multicollinearity
vif_values <- vif(step_model)
print(vif_values)

# --------
# Fit model to predict positive symptoms
formula <- as.formula(paste("TLI_DISORG_centered ~", paste(mean_centered_vars, collapse = " + ")))
full_model <- lm(formula, data = d1)

step_model <- step(full_model, direction = "backward")
summary(step_model)

# Check for multicollinearity
vif_values <- vif(step_model)
print(vif_values)


# --------------
# Use regularized regression to select important features
mean_centered_vars <- d1 %>%
  select(ends_with("_centered")) %>%
  select(-c(TLI_DISORG_centered, TLI_IMPOV_centered, nword_centered, ncontent_centered,nrepeated_centered)) %>%
  names()

#-----
# negative symptoms
# Prepare the design matrix excluding unwanted columns
X <- model.matrix(as.formula(paste("~", paste(mean_centered_vars, collapse = " + "))), data = d1)[, -1]
y <- d1$TLI_IMPOV_centered

# Fit LASSO model with cross-validation
lasso_model <- cv.glmnet(X, y, alpha = 1)

# Best lambda value
best_lambda <- lasso_model$lambda.min
print(best_lambda)

# Coefficients of the best model
lasso_coef <- coef(lasso_model, s = best_lambda)

# Extract and rank the feature names based on the absolute values of their coefficients
coef_df <- as.data.frame(as.matrix(lasso_coef))
coef_df <- coef_df[-1, , drop = FALSE]  # Remove intercept
colnames(coef_df) <- "Coefficient"
coef_df$Feature <- rownames(coef_df)
coef_df$Abs_Coefficient <- abs(coef_df$Coefficient)
coef_df <- coef_df[order(-coef_df$Abs_Coefficient), ]

# Calculate the R-squared value
predictions <- predict(lasso_model, X, s = best_lambda)
sse <- sum((y - predictions) ^ 2)
sst <- sum((y - mean(y)) ^ 2)
r_squared <- 1 - sse / sst

# Plot the coefficients
ggplot(coef_df, aes(x = reorder(Feature, Abs_Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "LASSO Regression for IMPOV",
       subtitle = paste("Explained Variance (R-squared):", round(r_squared, 2)),
       x = "Features",
       y = "Coefficient") +
  theme_minimal()



#-----
# positive symptoms
# Prepare the design matrix excluding unwanted columns
X <- model.matrix(as.formula(paste("~", paste(mean_centered_vars, collapse = " + "))), data = d1)[, -1]
y <- d1$TLI_DISORG_centered

# Fit LASSO model with cross-validation
lasso_model <- cv.glmnet(X, y, alpha = 1)

# Best lambda value
best_lambda <- lasso_model$lambda.min
print(best_lambda)

# Coefficients of the best model
lasso_coef <- coef(lasso_model, s = best_lambda)

# Extract and rank the feature names based on the absolute values of their coefficients
coef_df <- as.data.frame(as.matrix(lasso_coef))
coef_df <- coef_df[-1, , drop = FALSE]  # Remove intercept
colnames(coef_df) <- "Coefficient"
coef_df$Feature <- rownames(coef_df)
coef_df$Abs_Coefficient <- abs(coef_df$Coefficient)
coef_df <- coef_df[order(-coef_df$Abs_Coefficient), ]

# Calculate the R-squared value
predictions <- predict(lasso_model, X, s = best_lambda)
sse <- sum((y - predictions) ^ 2)
sst <- sum((y - mean(y)) ^ 2)
r_squared <- 1 - sse / sst

# Plot the coefficients
ggplot(coef_df, aes(x = reorder(Feature, Abs_Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "LASSO Regression for DISORG",
       subtitle = paste("Explained Variance (R-squared):", round(r_squared, 2)),
       x = "Features",
       y = "Coefficient") +
  theme_minimal()
