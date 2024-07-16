library(survival)
library(timeROC)
library(ggplot2)


compute_auc <- function(data, cohort_name) {
  # Prepare the data
  data$SCORE_BINARY <- ifelse(data$SCORE >= 0.9357855, 'ZYADA', 'KUM')
  data$SCORE_BINARY_NUM <- as.numeric(factor(data$SCORE_BINARY))
  data$DFS_YEARS <- data$DFS / 365.25

  # Compute time-dependent ROC curve at specific times (in years)
  roc_curve <- timeROC(T = data$DFS_YEARS,
                       delta = data$DFS_E,
                       marker = data$SCORE_BINARY_NUM,
                       cause = 1,
                       weighting = "cox",
                       times = c(2.5, 5.0, 7.5, 10.0))

  # Extract AUC values
  times <- c(0.5, 1.0, 1.5, 2.0, 2.5)
  auc_values <- roc_curve$AUC

  # Create a data frame for plotting
  plot_data <- data.frame(Time = times, AUC = auc_values, Cohort = cohort_name)
  return(plot_data)
}

train_cohort <- read.csv('/Users/vidhyasainath/Desktop/khooj/CIRCULATE/FINAL-DATASET/DACHS/TRAIN_score.csv')
valid_cohort <- read.csv('/Users/vidhyasainath/Desktop/khooj/CIRCULATE/FINAL-DATASET/DACHS/valid_score.csv')
test_cohort <- read.csv('/Users/vidhyasainath/Desktop/khooj/CIRCULATE/FINAL-DATASET/DACHS/TEST_score.csv')
circulate_cohort <- read.csv('/Users/vidhyasainath/Desktop/khooj/CIRCULATE/FINAL-DATASET/DACHS/CIRCULATE_score.csv')

plot_train <- compute_auc(train_cohort,"DACHS training set")
plot_valid <- compute_auc(valid_cohort,"DACHS validation set")
plot_test <- compute_auc(test_cohort,"DACHS test set")
plot_circulate <- compute_auc(circulate_cohort,"CIRCULATE set")

combined_plot_data <- rbind(plot_train, plot_valid, plot_test, plot_circulate)

ggplot(combined_plot_data, aes(x = Time, y = AUC, color = Cohort, group = Cohort)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed") + # Add horizontal line at y=0
  geom_vline(xintercept = 0, color = "black", linetype = "dashed") + # Add vertical line at x=0
  labs(title = "Time-Dependent ROC Curve for DACHS Cohorts",
       x = "Years",
       y = "AUC") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

ggsave("/Users/vidhyasainath/Desktop/khooj/CIRCULATE/FINAL-DATASET/DACHS/OVERALL_time_dependent_roc_curve_multiple_cohorts.svg")
