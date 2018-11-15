library(tidyverse)
library(data.table)
library(h2o)
library(caret)

digit <- fread("data/train.csv") %>%
	mutate(label = as.factor(label))

# Quick and dirty data augmentation
digit2 <- digit
digit2[,-1] <- digit2[,-1]/pi
digit <- rbind(digit,digit2)

digit3 <- digit
digit3[,-1] <- sqrt(digit3[,-1]/255)*255
digit <- rbind(digit,digit3)

rm(digit2,digit3);gc()


n_instances <- nrow(digit)

m <- sample(seq(from = 1, to = n_instances), n_instances*0.8, replace = FALSE)
train <- digit[m,]
val   <- digit[-m,]
h2o_train <- as.h2o(train)
h2o_val   <- as.h2o(val)

neurons_layer <- 300

h2o_model <- h2o.deeplearning(x = setdiff(names(train), c("label")),
															y = "label",
															training_frame = h2o_train,
															standardize = TRUE,
															hidden = c(neurons_layer,neurons_layer,neurons_layer),
															rate = 0.025,
															epochs = 15)

predictions_val   <- as.data.frame(h2o.predict(h2o_model, h2o_val))
predictions_train <- as.data.frame(h2o.predict(h2o_model, h2o_train))
acc_tr            <- confusionMatrix(predictions_train$predict, train$label) %>% .$overall %>% .["Accuracy"] %>% .[[1]]
acc_val           <- confusionMatrix(predictions_val$predict, val$label) %>% .$overall %>% .["Accuracy"] %>% .[[1]]

test <- fread("data/test.csv")
h2o_test <- as.h2o(test)
predictions_test <- as.data.frame(h2o.predict(h2o_model, h2o_test))
sub <- data.frame(ImageId = 1:nrow(test), Label = predictions_test$predict)

write.csv(sub, file = "submission-h2o.csv", row.names = F)
