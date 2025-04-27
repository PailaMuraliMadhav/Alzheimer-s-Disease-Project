
library(caret)
library(e1071)      # For Naive Bayes and SVM
library(class)      # For KNN
library(rpart)      # For Decision Tree
library(neuralnet)       # For Multiple Linear Regression
library(ggplot2)  

setwd("C:/Users/mural/OneDrive/Desktop/SEM 5/INT234")
data <- read.csv("alzheimers_disease_data.csv")

data <- data[, !(names(data) %in% c("PatientID", "DoctorInCharge"))]

data$Diagnosis <- as.factor(data$Diagnosis)

set.seed(123)
trainIndex <- createDataPartition(data$Diagnosis, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# KNN
knn_pred <- knn(train = trainData[, -ncol(trainData)], test = testData[, -ncol(testData)], 
                cl = trainData$Diagnosis, k = 5)
knn_acc <- sum(knn_pred == testData$Diagnosis) / length(testData$Diagnosis) * 100

# NB
nb_model <- naiveBayes(Diagnosis ~ ., data = trainData)
bnb_pred <- predict(nb_model, testData)
nb_acc <- sum(bnb_pred == testData$Diagnosis) / nrow(testData) * 100

# Decision Tree
dt_model <- rpart(Diagnosis ~ ., data = trainData, method = "class")
dt_pred <- predict(dt_model, testData, type = "class")
dt_acc <- sum(dt_pred == testData$Diagnosis) / nrow(testData) * 100

library(rpart.plot)  

rpart.plot(dt_model, 
           type = 2, 
           extra = 104, 
           fallen.leaves = TRUE, 
           main = "Decision Tree for Alzheimer's Diagnosis")

# SVM
svm_model <- svm(Diagnosis ~ ., data = trainData, kernel = "linear")
svm_pred <- predict(svm_model, testData)
svm_acc <- sum(svm_pred == testData$Diagnosis) / nrow(testData) * 100

# Multiple Linear Regression 
mlr_model <- multinom(Diagnosis ~ ., data = trainData)
mlr_pred <- predict(mlr_model, testData)
mlr_acc <- sum(mlr_pred == testData$Diagnosis) / nrow(testData) * 100

# Accuracy
cat("KNN Accuracy:", knn_acc, "%\n")
cat("Naive Bayes Accuracy:", nb_acc, "%\n")
cat("Decision Tree Accuracy:", dt_acc, "%\n")
cat("SVM Accuracy:", svm_acc, "%\n")
cat("Multiple Linear Regression Accuracy:", mlr_acc, "%\n")

accuracy_table <- data.frame(
  Model = c("KNN", "Naive Bayes", "Decision Tree", "SVM", "Multiple Linear Regression"),
  Accuracy = c(knn_acc, nb_acc, dt_acc, svm_acc, mlr_acc)
)

print(accuracy_table)

ggplot(accuracy_table, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(Accuracy, 2)), vjust = -0.5, size = 5) +
  labs(title = "Model Accuracy Comparison", y = "Accuracy (%)", x = "Model") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "none",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14))
