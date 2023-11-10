# Libraries----
library(tidyverse)
library(parsnip)
library(tidymodels)
library(yardstick)
library(ggcorrplot)
library(tune)

#Sampling data
library(rsample)

##1.0 Import data----
train <- read_csv("00_data/train.csv")
train
test <- read_csv("00_data/test.csv")
test
sample <- read_csv("00_data/submission_example.csv")

## 1.1 Visualize data and diagnosis----
### Correlation anlysis
### Correlation_plot

corr_matrix <- function(data){
    round(cor(data),1)
}

# corr_matrix <- round(cor(train %>% select(-ID)),1)
corr_matrix_plot <- function(corr_matrix) {
    corr_matrix%>% 
        ggcorrplot(method="circle", type="upper",
                   colors=c("red","white","red"),
                   hc.order = TRUE,
                   lab=TRUE, lab_size=2,
                   sig.level=0.05,
                   insig="blank") 
}
   
train %>% select(-ID) %>% 
    corr_matrix() %>% 
    corr_matrix_plot()

#Drop highly correlated features
train %>% select(-tax,-age, -nox, -ID) %>% 
    corr_matrix() %>% 
    corr_matrix_plot()

# Outlier detection
detect_outliers <- function(x){
    if(missing(x)) stop("The arg x needs a vector.")
    if (!is.numeric(x)) stop("The arg x must be numeric.")
    data_tbl <- tibble(data=x)
    limits_tbl <- data_tbl %>% 
        summarise(
            quantile_lo=quantile(data,probs=0.25, na.rm=TRUE),
            quantile_hi=quantile(data,probs=0.75,na.rm=TRUE),
            iqr        =IQR(data,na.rm=TRUE),
            limit_lo   =quantile_lo-1.5*iqr,
            limit_hi   =quantile_lo+1.5*iqr
        )
    output_tbl <- data_tbl %>% 
        mutate(outlier=case_when(
            data<limits_tbl$limit_lo~TRUE,
            data>limits_tbl$limit_hi~TRUE,
            TRUE~FALSE
        ))
    return(output_tbl$outlier)
}

train %>% 
    select(-ID) %>% 
    pivot_longer(-medv) %>% 
    group_by(name) %>% 
    mutate(outlier_value=value %>% detect_outliers()) %>% 
    ggplot(aes(medv,value))+
    geom_point(aes(color=outlier_value))+
    facet_wrap(~name)
#* Black zn and em seem to have multiple outliers

#Split data/sample
split_obj <- train %>% rsample::initial_split(prop=0.8,strata="rad")
    

train_tbl <- training(split_obj)
test_tbl <- testing(split_obj)
outcome <- "medv"

## 2.0 Recipe Pipeline----
recipe <- train_tbl %>% dplyr::select(-ID) %>% 
    recipe(reformulate(".",response = outcome)) %>% 
    step_zv(all_predictors()) %>% 
    # Normalize data
    step_scale(all_predictors()) %>% 
    step_center(all_predictors()) %>% 
    step_impute_mean(all_predictors()) %>% 
    prep() 

train_tbl <- recipe %>% bake(train_tbl)

test_tbl <- recipe %>% bake(test_tbl)


## Fit linear regression model
set.seed(123)
#Linear regression
outcome <- "medv"

model_lm <- linear_reg(mode="regression") %>% 
    set_engine("lm") %>% 
    fit(reformulate(".",response = outcome),
        data=train_tbl %>% select(-tax,-age, -nox))

## Metric Calculation function

calc_metrics <- function(model,new_data=test_data) {
    model %>% 
        predict(new_data=new_data) %>% 
        bind_cols(new_data %>% select(outcome)) %>% 
        yardstick::metrics(truth=outcome,estimate=.pred)
}


model_lm %>% calc_metrics(new_data = test_tbl %>% select(-tax,-age, -nox))

## Predict using linear model
model_lm1 <- linear_reg(mode="regression") %>% 
    set_engine("lm") %>% 
    fit(reformulate(".",response = outcome),
        data=train %>% select(-ID,-tax,-age, -nox))

predict.model_fit(model_lm,test) %>% 
    bind_cols(test %>% select(ID)) %>% 
    select(ID,medv=.pred) %>% 
    write_csv("01_outputs/submission_file.csv")

## X_gboost model tunning

set.seed(100)
xgb_spec <- boost_tree(mode="regression",
                             trees=tune(),
                             learn_rate = 0.02) %>% 
    set_engine("xgboost",lambda=tune()) 

lambda <- function(range = c(-10, 0), trans = log10_trans()) {
    new_quant_param(
        type = "double",
        range = range,
        inclusive = c(TRUE, TRUE),
        trans = trans,
        label = c(lambda = "Amount of Regularization"),
        finalize = NULL
    )
}

param_set <- parameters(list(lambda(), trees()))

folds <- vfold_cv(train_tbl %>% dplyr::select(-ID), v = 7)

xg_boost_model_tuned <- tune_grid(xgb_spec, 
          reformulate(".",response=outcome),
          resamples = folds, 
          param_info = param_set)

#Extract best models post tuning

best_xgb_model <- select_best(xg_boost_model_tuned,metric="rmse")

best_xgb_model

#Fit model with best model parameters
xg_boost_model <- boost_tree(mode="regression",
                             trees = 1212,
                             mtry  =3) %>% 
    set_engine("xgboost",lambda=0.123) %>% 
    fit(reformulate(".",response=outcome),
                    data=train %>% dplyr::select(-ID))

#xg_boost_model <- finalize_model(best_xgb_model)

#Calculate model metrics

xg_boost_model %>% calc_metrics(new_data = test_tbl)


#Predict
# xg_boost_model1 <- boost_tree(mode="regression",
#                               trees=150,
#                               mtry=3,
#                               learn_rate = 0.2) %>% 
#     set_engine("xgboost") %>% 
#     fit(reformulate(".",response=outcome), data=train_tbl)
# xg_boost_model1 %>% calc_metrics(new_data = test_tbl)


## 3.0 H2O Modelling----

library(h2o)
library(tidyquant)

h2o.init()

setSessionTimeLimit(elapsed = Inf)

split_obj_h2o <- h2o.splitFrame(as.h2o(train_tbl),ratios = c(0.75))

train_h2o <- split_obj_h2o[[1]]
valid_h2o <- split_obj_h2o[[2]]

test_h2o <- as.h2o(test_tbl)

y <- "medv"
x <- setdiff(names(train_h2o),y)

h2o_models <- h2o.automl(x=x,y=y,
           training_frame = train_h2o,
           validation_frame = valid_h2o,
           leaderboard_frame = test_h2o,
           nfolds = 10,
           max_runtime_secs = 30)

h2o_leaderboard <- h2o_models@leaderboard

#Visualize H2O model leaderboard

plot_h2o_leaderboard <- function(h2o_leaderboard,order_by=c("rmse","mean_residual_deviance"),
                                 n_max=20,size=4,include_lbl=TRUE){
    
    order_by <- tolower(order_by[[1]])
    
    leaderboard_tbl <- h2o_leaderboard%>% 
        as.tibble() %>% 
        mutate(model_type=str_split(model_id,"_",simplify = T)[,1]) %>% 
        rownames_to_column(var="rowname") %>% 
        mutate(model_id=paste0(rowname,". ",as.character(model_id)) %>% as_factor())
    
    #transformation
    
    if(order_by=="rmse"){
        data_transformed_tbl <- leaderboard_tbl %>% 
            slice(1:n_max) %>% 
            mutate(
                model_id  =as_factor(model_id) %>% reorder(rmse),
                model_type=as.factor(model_type)
            ) %>% 
            gather(key=key,value=value,c(rmse,mean_residual_deviance),factor_key = T)
    } else if (order_by=="mean_residual_deviance"){
        data_transformed_tbl <- leaderboard_tbl %>% 
            slice(1:n_max) %>% 
            mutate(
                model_id  =as_factor(model_id) %>% reorder(mean_residual_deviance) %>% fct_rev(),
                model_type=as.factor(model_type)
            ) %>% 
            gather(key=key,value=value,c(rmse,mean_residual_deviance),factor_key = T)       
        
    } else {
        stop(paste0("order_by",order_by,"'is not a permitted option."))
    }
    
    #Visualization
    
    g <- data_transformed_tbl %>% 
        ggplot(aes(value,model_id, color=model_type))+
        geom_point(size=size)+
        facet_wrap(~key,scales="free_x")+
        theme_tq()+
        scale_color_tq()+
        labs(
            title= "Leaderboard Metrics",
            subtitle =paste0("Ordered by: ", toupper(order_by)),
            y="Model position, Model ID", x="")
    
    if(include_lbl) g <- g+geom_label(aes(label=round(value,2),hjust="inward"))
    
    return(g)
}


h2o_models@leaderboard %>% plot_h2o_leaderboard(order_by = "rmse")



# h2o.getModel("DeepLearning_grid_1_AutoML_2_20230904_90111_model_2") %>% 
#     h2o.saveModel(path="./02_Model")

# deeplearning_model <- h2o.loadModel("02_Model/DeepLearning_grid_1_AutoML_2_20230904_90111_model_2")

h2o_best_model <- h2o.get_best_model(h2o_leaderboard[[1]] %>% as.vector() %>% pluck(1))

#Predict 
predict.model_fit(xg_boost_model,test) %>% 
    bind_cols(test %>% select(ID)) %>% 
    select(ID,medv=.pred) %>% 
    write_csv(str_glue("01_outputs/submission_file_{today()}.csv"))

h2o_predicted <- h2o.predict(h2o_models,newdata = as.h2o(test)) %>% 
    as_tibble() %>% 
    bind_cols(test %>% select(ID)) %>% 
    select(ID,medv=predict)

write_csv(str_glue(h2o_predicted,"01_outputs/submission_file_{today()}.csv"))
    

