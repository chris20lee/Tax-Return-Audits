# Install packages if not already installed
if(!require(caret)) install.packages("caret") # Dummy variables
if(!require(class)) install.packages("class") # K-Nearest Neighbour
if(!require(e1071)) install.packages("e1071") # Naive Bayes
if(!require(FactoMineR)) install.packages("FactoMineR") # MCA
if(!require(factoextra)) install.packages("factoextra") # Graph MCA
if(!require(missForest)) install.packages("missForest") # Impute missing values
if(!require(nnet)) install.packages("nnet") # Neural networks
if(!require(randomForest)) install.packages("randomForest") # Random Forest
if(!require(rggobi)) install.packages("rggobi") # ggobi
if(!require(stats)) install.packages("stats") # K-Mmeans

# Load dataset into R
file_dir = paste0("C:/Users/Chris/Desktop/audit.csv")
audit = read.csv(file = file_dir)

# View dataset properties before data manipulation
dim(audit)
sapply(audit,class)
(sapply(audit, function(x) sum(length(which(is.na(x))))))
summary(audit)

# DATA REDUCTION

# Drop the ID variable
audit = subset(audit, select = -ID)

# Set the one record unemployed for both occupation and unemployment
audit$Occupation = as.character(audit$Occupation)
audit$Occupation[audit$Employment == "Unemployed"] = "Unemployed"
audit$Occupation = as.factor(audit$Occupation)

# Impute missing values, using all parameters as default values
imp = missForest(audit, maxiter = 10)
audit_data = imp$ximp

# Visualize data in ggobi
audit_vis = ggobi(audit)
display(audit_vis[1], "Scatterplot Matrix")
display(audit_vis[1], "Parallel Coordinates Display")
close(audit_vis)

# Random Forest to determine feature selection
feature_selection = randomForest(as.factor(TARGET_Adjusted) ~ Age + Employment + Education + Marital + Occupation + Income + 
                       Gender + Deductions + Hours, data = audit_data, ntree = 2000, replace = TRUE, importance = TRUE)
feature_selection
varImpPlot(feature_selection)
imp_vars = as.data.frame(data.frame(importance(feature_selection)))
imp_vars = subset(imp_vars, select = MeanDecreaseAccuracy)
imp_vars$perc = apply(imp_vars, 2, function(x) x/sum(x))
imp_vars = imp_vars[order(imp_vars$perc),]
imp_vars$cum_perc = cumsum(imp_vars$perc)
# Drop variables that make up less than 10% of the variance
imp_vars = imp_vars[which(imp_vars$cum_perc >= 0.1),]
sel_vars = append(rownames(imp_vars), c("RISK_Adjustment", "TARGET_Adjusted"))

# Save new reduced dataset
audit_data = subset(audit_data, select = sel_vars)

# VISUALIZATION

# Visualize variable stats
for (i in 1:ncol(audit_data)) {
  print(table(audit_data[i]))
  plot(audit_data[,i], main=colnames(audit_data)[i], ylab = "Count")
  readline("Hit Enter for next column: ")
}

# View dataset properties after data manipulation
dim(audit_data)
sapply(audit_data,class)
(sapply(audit_data, function(x) sum(length(which(is.na(x))))))
summary(audit_data)

# DIMENSION REDUCTION

# Create dummy variables
dmy = dummyVars("~ .", data = audit_data)
audit_data_dum = data.frame(predict(dmy, newdata = audit_data))

#Normalize data
audit_data_dum_norm = as.data.frame(apply(audit_data_dum[,], 2, function(x) (x - min(x))/(max(x)-min(x))))

# PCA on normalized data
audit_pca_dum_norm = prcomp(audit_data_dum_norm)
summary(audit_pca_dum_norm)
audit_pca_dum_norm
fviz_screeplot(audit_pca_dum_norm, addlabels = TRUE)

# MCA
audit_data_factor = audit_data[,sapply(audit_data, is.factor)]
audit_mca = MCA(audit_data_factor, graph = FALSE)
eig_val = get_eigenvalue(audit_mca)
fviz_screeplot(audit_mca, addlabels = TRUE)

# PCA and MCA
together = FAMD(audit_data, ncp = ncol(audit_data))
summary(together)

# UNSUPERVISED LEARNING (CLUSTERING)

# K-Means cluster
KM = kmeans(audit_data_dum_norm[1:(ncol(audit_data_dum_norm) - 2)], 2, 10)
table(Predicted = KM$cluster, Actual = audit_data$TARGET_Adjusted)

# Hierarchical clustering
c_dist = dist(audit_data_dum)

# Single linkage hierarchical clustering
c_hclust = hclust(c_dist, method = "single" )
c_hclust$labels = audit_data_dum$TARGET_Adjusted
plot(c_hclust, main = "Single Linkage", cex = 0.6)
rect.hclust(c_hclust, k = 2, border = "red")

# Complete linkage hierarchical clustering
c_hclust = hclust(c_dist, method = "complete" )
c_hclust$labels = audit_data_dum$TARGET_Adjusted
plot(c_hclust, main = "Complete Linkage", cex = 0.6)
rect.hclust(c_hclust, k = 2, border = "red")

# Average linkage hierarchical clustering
c_hclust=hclust(c_dist, method = "ave" )
c_hclust$labels = audit_data_dum$TARGET_Adjusted
plot(c_hclust, main = "Average Linkage", cex = 0.6)
rect.hclust(c_hclust, k = 2, border = "red")

# SUPERVISED LEARNING (CLASSIFICATION)

# Create training and testing sets
set.seed(123)
train = sample(nrow(audit_data), 0.8 * nrow(audit_data), replace = TRUE)
# Train and test sets for prediction of RISK_Adjustment
train_set = audit_data[train,]
test_set = audit_data[-train,]
# Train and test sets for prediction of TARGET_Adjusted
train_set_no_risk = subset(train_set, select = - RISK_Adjustment)
test_set_no_risk = subset(test_set, select = - RISK_Adjustment)

# Random Forest on TARGET_Adjusted
rf = randomForest(as.factor(TARGET_Adjusted) ~ ., data = train_set_no_risk, replace = TRUE, importance = TRUE)
rf
plot(rf, main = "Random Forest")
# Prediction on the training set
pred_train_rf = predict(rf, train_set_no_risk, type = "class")
table(Predicted = pred_train_rf, Actual = train_set_no_risk$TARGET_Adjusted)
# Prediction on the test set
pred_test_rf = predict(rf, test_set_no_risk, type = "class")
table(Predicted = pred_test_rf, Actual = test_set_no_risk$TARGET_Adjusted)
mean(pred_test_rf == test_set_no_risk$TARGET_Adjusted) 

# K-NN on TARGET_Adjusted
train_x_knn = model.matrix(~ 0 + ., data = train_set_no_risk)
test_x_knn = model.matrix(~ 0 + ., data = test_set_no_risk)
train_y_knn = train_set_no_risk$TARGET_Adjusted
# Prediction on the test set
test_y_knn = knn(train_x_knn, test_x_knn, train_y_knn, k=5)
table(Predicted = test_y_knn, Actual = test_set_no_risk$TARGET_Adjusted)
mean(test_y_knn == test_set_no_risk$TARGET_Adjusted) 

# Naive Bayes on TARGET_Adjusted
nb = naiveBayes(as.factor(TARGET_Adjusted) ~ ., data=train_set_no_risk)
# Prediction on the training set
pred_train_nb = predict(nb, train_set_no_risk)
table(Predicted = pred_train_nb, Actual = train_set_no_risk$TARGET_Adjusted)
# Prediction on the test set
pred_test_nb = predict(nb, test_set_no_risk)
table(Predicted = pred_test_nb, Actual = test_set_no_risk$TARGET_Adjusted)
mean(pred_test_nb == test_set_no_risk$TARGET_Adjusted) 

# Neural Networks on TARGET_Adjusted
nn = nnet(as.factor(TARGET_Adjusted) ~ ., data = train_set_no_risk, size = 5)
summary(nn)
# Prediction on the training set
pred_train_nn = predict(nn, newdata = train_set_no_risk, type = "class")
table(Predicted = pred_train_nn, Actual = train_set_no_risk$TARGET_Adjusted)
# Prediction on the test set
pred_test_nn = predict(nn, newdata = test_set_no_risk, type = "class")
table(Predicted = pred_test_nn, Actual = test_set_no_risk$TARGET_Adjusted)
mean(pred_test_nn == test_set_no_risk$TARGET_Adjusted) 

# Attempt to do model averaging ensembles but for some reason the averaging gives wrong results mathematically
test_results = subset(test_set, select = TARGET_Adjusted)
test_results$pred_rf_prob = predict(rf, test_set_no_risk, type = "class")
test_results$pred_knn_prob = knn(train_x_knn, test_x_knn, train_y_knn, k=5)
test_results$pred_nb_prob = predict(nb, test_set_no_risk)
test_results$pred_nn_prob = predict(nn, newdata = test_set_no_risk, type = "class")
test_results$pred_avg = round((as.numeric(test_results$pred_rf_prob) + as.numeric(test_results$pred_knn_prob) + 
                       as.numeric(test_results$pred_nb_prob) + as.numeric(test_results$pred_nn_prob))/4,0)
table(Predicted = test_results$pred_avg, Actual = test_results$TARGET_Adjusted)

# Random Forest on RISK_Adjustment
model1 = randomForest(RISK_Adjustment ~ ., data = train_set, replace = TRUE, importance = TRUE)
model1
plot(model1)
plot(train_set$RISK_Adjustment, train_set$RISK_Adjustment - model1$predicted, main = "Residuals")

# Regression on RISK_Adjustment
reg = lm(RISK_Adjustment ~ ., data = train_set)
summary(reg)
pred = predict(reg, train_set)
pred = as.data.frame(pred)
x = as.data.frame(resid(reg))
plot(density(resid(reg)))
plot(train_set$RISK_Adjustment, resid(reg), main = "Residuals")
qqnorm(resid(reg))
qqline(resid(reg))

graphics.off()
