library(tidyverse)
library(h2o)
library(lime)

data <- read.csv("winemag-data_first150k.csv")

# Remove X index column
data <- data %>% select(-X, -description)

data %>% as_tibble()

str(data)


# Set seed for reproducability
n_seed = 12345


### Predicting points -----------------------------------------


# Create target and feature list
#target = "points" # Result
target = "price" # Result
features = setdiff(colnames(data), target)
print(features)


# Start a local H2O cluster (JVM)
h2o.init()

# H2O dataframe
h_data <-  as.h2o(data)


# Split Train/Test
h_split = h2o.splitFrame(h_data, ratios = 0.75, seed = n_seed)
h_train = h_split[[1]] # 75% for modelling
h_test = h_split[[2]] # 25% for evaluation


model_glm = h2o.glm(x = features,
                    y = target,
                    nfolds = 5,
                    training_frame = h_train,
                    model_id = "my_glm",
                    standardize = TRUE,
                    seed = n_seed)
print(model_glm)


# Evaluate performance on test
h2o.performance(model_glm, newdata = h_test)



### Test other algos with AutoML

model_automl <- h2o.automl(x = features,
                          y = target,
                          training_frame = h_train,
                          nfolds = 5,               # Cross-Validation
                          max_runtime_secs = 30,   # Max time
                          max_models = 100,         # Max no. of models
                          stopping_metric = "RMSE", # Metric to optimize
                          project_name = "my_automl",
                          exclude_algos = NULL,     # If you want to exclude any algo 
                          seed = n_seed)


model_automl@leaderboard 

model_automl@leader

h2o.performance(model_automl@leader, newdata = h_test)



explainer = lime(x = as.data.frame(h_train[, features]),model = model_automl@leader)

# Extract one sample (change `1` to any row you want)
d_samp = as.data.frame(h_test[1, features])
# Assign a specifc row name (for better visualization)
row.names(d_samp) = "Sample 1" 
# Create explanations
explanations = lime::explain(x = d_samp,
                             explainer = explainer,
                             n_permutations = 5000,
                             feature_select = "auto",
                             n_features = 13) # Look top x features

lime::plot_features(explanations, ncol = 1)







