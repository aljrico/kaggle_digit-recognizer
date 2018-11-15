library(tidyverse)
library(data.table)
library(caret)
library(keras)

moveme <- function (invec, movecommand) {
	movecommand <- lapply(strsplit(strsplit(movecommand, ";")[[1]],
																 ",|\\s+"), function(x) x[x != ""])
	movelist <- lapply(movecommand, function(x) {
		Where <- x[which(x %in% c("before", "after", "first",
															"last")):length(x)]
		ToMove <- setdiff(x, Where)
		list(ToMove, Where)
	})
	myVec <- invec
	for (i in seq_along(movelist)) {
		temp <- setdiff(myVec, movelist[[i]][[1]])
		A <- movelist[[i]][[2]][1]
		if (A %in% c("before", "after")) {
			ba <- movelist[[i]][[2]][2]
			if (A == "before") {
				after <- match(ba, temp) - 1
			}
			else if (A == "after") {
				after <- match(ba, temp)
			}
		}
		else if (A == "first") {
			after <- 0
		}
		else if (A == "last") {
			after <- length(myVec)
		}
		myVec <- append(temp, values = movelist[[i]][[1]], after = after)
	}
	myVec
}

# Preprocessing Data ---------------------------------------------------------------
train <- fread("data/train.csv")
test  <- fread("data/test.csv") %>% data.matrix()

digit7 <- train[label == 7, -1] %>% data.frame()
digit1 <- train[label == 1, -1] %>% data.frame()

# Quick and dirty data augmentation
digit2 <- digit7
digit2 <- digit2/pi
digit7 <- rbind(digit7,digit2)

digit3 <- digit7
digit3 <- sqrt(digit3/255)*255
digit7 <- rbind(digit7,digit3)
digit7$label <- 7

rm(digit2,digit3);gc()

digit2 <- digit1
digit2 <- digit2/pi
digit1 <- rbind(digit1,digit2)

digit3 <- digit1
digit3 <- sqrt(digit3/255)*255
digit1 <- rbind(digit1,digit3)
digit1$label <- 1

digit <- rbind(digit1,digit7)
digit[moveme(names(digit), "label first")]
rm(digit2,digit3,digit1,digit7);gc()

train <- rbind(train,digit) %>% data.matrix()

train_label <- train[,1] %>% data.matrix()  %>% to_categorical()
train_features <- train[,-1] %>% normalize()
test_features <- test %>% normalize()

dim(train_features)  <- c(nrow(train_features),28,28,1)
dim(test_features)   <- c(nrow(test_features),28,28,1)



# Model -------------------------------------------------------------------

model <- keras_model_sequential()

model %>%
	layer_conv_2d(filters = 32,
								kernel_size = c(5,5),
								padding = 'Valid',
								activation = 'relu',
								input_shape = c(28,28,1)) %>%
	layer_batch_normalization() %>%
	layer_conv_2d(filters = 32,
								kernel_size = c(5,5),
								padding = 'Same',
								activation = 'relu') %>%
	layer_batch_normalization() %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	layer_dropout(rate = 0.2) %>%
	layer_conv_2d(filters = 64,
								kernel_size = c(3,3),
								padding = 'Same',
								activation = 'relu') %>%
	layer_batch_normalization() %>%
	layer_conv_2d(filters = 64,
								kernel_size = c(3,3),
								padding = 'Same',
								activation = 'relu') %>%
	layer_batch_normalization() %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	layer_dropout(rate = 0.2) %>%
	layer_flatten() %>%
	layer_dense(units = 1024, activation = 'relu') %>%
	layer_dense(units = 512,  activation = 'relu') %>%
	layer_dense(units = 256,  activation = 'relu') %>%
	layer_dense(units = 128,  activation = 'relu') %>%
	layer_dense(units = 64,   activation = 'relu') %>%
	layer_dense(units = 32,   activation = 'relu') %>%
	layer_dense(units = 16,   activation = 'relu') %>%
	layer_dense(units = 10,   activation = 'softmax')

model %>% compile(
	loss = 'categorical_crossentropy',
	optimizer = 'adam',
	metrics   = 'accuracy'
)

datagen <- image_data_generator(
	featurewise_center = FALSE,
	samplewise_center  = FALSE,
	featurewise_std_normalization = FALSE,
	samplewise_std_normalization = FALSE,
	zca_whitening = FALSE,
	horizontal_flip = FALSE,
	vertical_flip   = FALSE,
	width_shift_range = 0.15,
	height_shift_range = 0.15,
	zoom_range = 0.15,
	rotation_range = 0.15,
	shear_range = 0.15
)

datagen %>% fit_image_data_generator(train_features)

history <- model %>%
	fit_generator(flow_images_from_data(train_features,
																			train_label,
																			datagen,
																			batch_size = 64),
								steps_per_epoch = nrow(train_features)/64, epochs = 10)

plot(history)

pred <- model %>% predict_classes(test_features, batch_size = 64)

cnn_submission <- data.frame(ImageId = 1:nrow(test),
														 Label = pred)

write.csv(cnn_submission, file="cnn_submission.csv", row.names=F)
