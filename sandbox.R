library(tidyverse)
library(data.table)
library(h2o)
library(caret)
library(dataPreparation)
library(tictoc)
library(harrypotter)
library(gridExtra)
library(plotly)


digit <- fread("data/train.csv")
table(digit$label)/length(digit$label)

digit$label %>% table() %>% prop.table()
digit$label <- digit$label %>% factor()

whichAreConstant(digit,verbose=TRUE)
constant_columns <- whichAreConstant(digit, verbose=FALSE)
if(length(constant_columns > 0)) digit <- subset(digit,select = -c(constant_columns)) %>% as_tibble()

n_iters <- 90
n_instances <- nrow(digit)
acc <- c()
time <- c()

tic()
for(i in 1:n_iters){
	m <- sample(seq(from = 1, to = n_instances), n_instances*0.9, replace = FALSE)
	train <- digit[m,]
	val   <- digit[-m,]
	h2o_train <- as.h2o(train)
	h2o_val   <- as.h2o(val)

	time[i] <- system.time({
		h2o_model <- h2o.deeplearning(x = setdiff(names(train), c("label")),
																y = "label",
																training_frame = h2o_train,
																standardize = TRUE,
																hidden = c(20 + i*2,10 + i,20 + i*2),
																rate = 0.05,
																epochs = 10)
	})[1][[1]]
	h2o_predictions <- as.data.frame(h2o.predict(h2o_model, h2o_val))
	print(acc[i] <- confusionMatrix(h2o_predictions$predict, val$label) %>% .$overall %>% .["Accuracy"] %>% .[[1]])
}

toc()

tibble(Accuracy = acc, Time = time, Iteration = 1:n_iters) %>%
	melt(id.vars = "Iteration") %>%
	ggplot(aes(x = 20 + 2*Iteration, y = value, colour = variable)) +
	geom_point()

tibble(Accuracy = acc, Time = time, Iteration = 1:n_iters) %>%
	ggplot(aes(x = Time, y = Accuracy)) +
	geom_jitter() +
	geom_smooth()

readRDS("df") %>%
	melt(id.vars = "Iteration") %>%
	ggplot(aes(x = 20 + 2*Iteration, y = value, colour = variable)) +
	geom_point()

gg_ite <- readRDS("df") %>%
	ggplot(aes(x = Iteration, y = Time)) +
	geom_jitter(colour = hp(10, house = "Ravenclaw")[[10]]) +
	geom_smooth(colour = hp(10, house = "Gryffindor")[[1]]) +
	xlab("Number of Neutrons") +
	ylab("Time (Minutes)")

gg_time <- readRDS("df") %>%
ggplot(aes(x = Time, y = Accuracy)) +
	geom_jitter(colour = hp(10, house = "Ravenclaw")[[10]]) +
	geom_smooth(colour = hp(10, house = "Ravenclaw")[[1]]) +
	xlab("Time (Minutes)")

gg_neurs <- readRDS("df") %>%
	ggplot(aes(x = 20 + 3*Iteration, y = Accuracy)) +
	geom_jitter(colour = hp(10, house = "Ravenclaw")[[10]]) +
	geom_smooth(colour = hp(10, house = "Slytherin")[[8]]) +
	xlab("Number of Neurons")

grid.arrange(gg_ite, gg_neurs, gg_time,
						 layout_matrix = rbind(c(1, 1),
						 											c(2, 3))
						 )

