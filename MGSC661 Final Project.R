# MGSC661 Final Project
# Anqi Chen 261044081
# Yulin Hong 260898713

######################################################################################
# Step 1: Data preprocessing
######################################################################################
# Import dataset
lol <- read.csv(file = 'final_match.csv')
attach(lol)

######################################################################################
# Step 2: Data exploration and feature selection
######################################################################################

# Boxplot and histogram of each individual variable
# Kill-deaths/assist ratio
par(mfrow=c(1,2))
boxplot(kda, main="boxplot of kill-death-assist ratio",col='#253B5C')
hist(kda, breaks=40, col='#253B5C', xlab="kill-death-assist ratio")

# Total damage dealt
boxplot(totdmgdealt, main="boxplot of Total damage dealt",col='#253B5C')
hist(totdmgdealt, breaks=40, col='#253B5C', xlab="Total damage dealt")

# Total damage to champion
boxplot(totdmgtochamp, main="boxplot of totdmgtochamp",col='blue')
hist(totdmgtochamp, breaks=40, col='blue', xlab="totdmgtochamp")

# Total damage taken
boxplot(totdmgtaken, main="boxplot of totdmgtaken",col='blue')
hist(totdmgtaken, breaks=40, col='blue', xlab="totdmgtaken")

# gold earned
boxplot(goldearned, main="boxplot of goldearned",col='blue')
hist(goldearned, breaks=40, col='blue', xlab="goldearned")

# gold spent
boxplot(goldspent, main="boxplot of goldspent",col='blue')
hist(goldspent, breaks=40, col='blue', xlab="goldspent")

# Total minion killed
boxplot(totminionskilled, main="boxplot of totminionskilled",col='blue')
hist(totminionskilled, breaks=40, col='blue', xlab="totminionskilled")

# neutral minion skilled
boxplot(neutralminionskilled, main="boxplot of neutralminionskilled",col='blue')
hist(neutralminionskilled, breaks=40, col='blue', xlab="neutralminionskilled")

# chanpion level
boxplot(champlvl, main="boxplot of champlvl",col='blue')
hist(champlvl, breaks=40, col='blue', xlab="champlvl")

# Tower kills
boxplot(towerkills, main="boxplot of towerkills",col='blue')
hist(towerkills, breaks=40, col='blue', xlab="towerkills")

# Inhibitor kills
boxplot(inhibkills, main="boxplot of inhibkills",col='blue')
hist(inhibkills, breaks=40, col='blue', xlab="inhibkills")

# Baron kills
boxplot(baronkills, main="boxplot of baronkills",col='blue')
hist(baronkills, breaks=40, col='blue', xlab="baronkills")

# Dragon kills
boxplot(dragonkills, main="boxplot of dragonkills",col='blue')
hist(dragonkills, breaks=40, col='blue', xlab="dragonkills")

# Harry kills
boxplot(harrykills, main="boxplot of harrykills",col='blue')
hist(harrykills, breaks=40, col='blue', xlab="harrykills")

# Duration
boxplot(duration, main="boxplot of duration",col='#253B5C')
hist(duration, breaks=40, col='#253B5C', xlab="duration")

par(mfrow=c(1,1)) # Reset

# Correlation 
continuous <- lol[,c(3,4,5,6,7,8,9,10,11,12,19,20,21,22,23,24)]
M <- round(cor(continuous), 2)
library(ggcorrplot)
ggcorrplot(M, lab = TRUE, outline.col = "white",
           ggtheme = ggplot2::theme_gray, colors = c("#BBEBC3", 'white', "#253B5C")) + 
  scale_x_discrete(labels = c('win', 'kda', 'total damage dealt', 'total damage to champ', 'total damage taken',
                              'gold earned', 'gold spent', 'total minion skilled', 'neutral minion skilled', 
                              'champ level', 'tower kills', 'inhibitor kills', 'baron kills', 'dragon kills',
                              'harry kills', 'duration')) + 
  scale_y_discrete(labels = c('win', 'kda', 'total damage dealt', 'total damage to champ', 'total damage taken',
                              'gold earned', 'gold spent', 'total minion skilled', 'neutral minion skilled', 
                              'champ level', 'tower kills', 'inhibitor kills', 'baron kills', 'dragon kills',
                              'harry kills', 'duration')) +
  labs(title = "Correlation Matrix")

# Remove IDs
drops <- c("matchid","teamid")
lol <- lol[ , !(names(lol) %in% drops)]
attach(lol)

# Principal component analysis
pca <- prcomp(na.omit(lol), scale = TRUE)
pca

# Plot the pca
# install.packages('ggfortify')
library(ggfortify)
autoplot(pca, data = na.omit(lol), loadings = TRUE, col = '#253B5C', loadings.label = TRUE, loadings.label.size= 4)

# For analysis purpose (analyze champlvl >= 15)
autoplot(pca, data = na.omit(lol), loadings = TRUE, col = ifelse(lol$champlvl >= 15, 'blue', 'transparent'), loadings.label = TRUE)

# To find percentage of variance explained
pve <- (pca$sdev^2)/sum(pca$sdev^2)
par(mfrow = c(1,2))
plot(pve, ylim = c(0,1))
plot(cumsum(pve), ylim = c(0,1))
pve

# Final predictors:
# kda, totdmgtochamp, totdmgtaken, champlvl, towerkills, inhibkills, duration, firstinhib, baronkills

######################################################################################
# Step 3: Classification models
######################################################################################
# Modify the target variable
lol$win <- ifelse(win == 0, 0, ifelse(win == 0.4, 0, ifelse(win == 0.6, 1, 1)))
table(lol$win)

# Create a dataframe to store accuracy score of each model
models <- c('Logistic regression', 'LDA', 'Decision Tree', 'Random Forest', 'Boosted Forest')
accuracy_score <- c()

###### Logistic regression ######
logit <- glm(win ~ kda + totdmgtochamp + totdmgtaken + champlvl + 
               towerkills + inhibkills + duration + firstinhib + 
               baronkills, data = lol, family = 'binomial')
summary(logit)

# To visualize with one predictor
plot <- ggplot(lol, aes(y = win, x = kda))
scatter = geom_point()
line = geom_smooth(method = 'glm', formula = y~x, method.args=list(family = binomial))
plot + scatter + line

# Just to see R2
require(rms)
mlogit <- lrm(win ~ kda + totdmgtochamp + totdmgtaken + champlvl + 
                towerkills + inhibkills + duration + firstinhib + 
                baronkills, data = lol)
mlogit

# Out-of-sample performance 
library(caTools)
mse = c()
accuracy = c()
for (i in 1:10){
  sample = sample.split(lol$win, SplitRatio = 0.75)
  train = subset(lol, sample == TRUE)
  test= subset(lol, sample == FALSE)
  # Build the model
  fit = glm(train$win ~ kda + totdmgtochamp + totdmgtaken + champlvl + 
              towerkills + inhibkills + duration + firstinhib + 
              baronkills, data=train, family = 'binomial')
  # MSE
  test$pred = predict(fit, test)
  test$res = (test$win - test$pred)
  test$res_sq = (test$res)^2
  MSE = mean(test$res_sq)
  mse[i] = MSE
  print(paste0('MSE = ', MSE))
  
  # Accuracy
  predicted_score <- predict(fit, test, type = 'response')
  result <- ifelse(predicted_score >= 0.5, 1, 0)
  print(paste0('Accuracy = ', mean(result == test$win)))
  accuracy[i] = mean(result == test$win)
}
mean(mse)
mean(accuracy)
accuracy_score <- c(accuracy_score, mean(accuracy))

###### Discriminant analysis ######
library(MASS)
library(klaR)
mylda <- lda(win ~ kda + totdmgtochamp + totdmgtaken + champlvl + 
               towerkills + inhibkills + duration + firstinhib + 
               baronkills)
mylda # linear

# To compare LDA and QDA error rate
cv.lda <-
  function (data, model= win~., yname="win", K=10, seed=100) {
    n <- nrow(data)
    set.seed(seed)
    datay=data[,win] #response variable
    #partition the data into K subsets
    f <- ceiling(n/K)
    s <- sample(rep(1:K, f), n)  
    #generate indices 1:10 and sample n of them  
    # K fold cross-validated error
    
    CV=NULL
    
    for (i in 1:K) { #i=1
      test.index <- seq_len(n)[(s == i)] #test data
      train.index <- seq_len(n)[(s != i)] #training data
      
      #model with training data
      lda.fit=lda(model, data=data[train.index,])
      #observed test set y
      lda.y <- data[test.index, yname]
      #predicted test set y
      lda.predy=predict(lda.fit, data[test.index,])$class
      
      #observed - predicted on test data
      error= mean(lda.y!=lda.predy)
      #error rates 
      CV=c(CV,error)
    }
    #Output
    list(call = model, K = K, 
         lda_error_rate = mean(CV), seed = seed)  
  }

er_lda=cv.lda(data=lol,model=win~kda + totdmgtochamp + totdmgtaken + champlvl + 
                towerkills + inhibkills + duration + firstinhib + 
                baronkills, yname="win", K=10, seed=100)
print(paste0('LDA accuracy rate = ', 1-er_lda$lda_error_rate))
accuracy_score <- c(accuracy_score, 1-er_lda$lda_error_rate)

######################################################################################
# Step 4: Tree based models
######################################################################################
###### Decision tree ######
library(tree)
library(rpart)
library(rpart.plot)

attach(lol)
mytree <- rpart(win~kda + totdmgtochamp + totdmgtaken + champlvl + 
                  towerkills + inhibkills + duration + firstinhib + 
                  baronkills, data = lol, control = rpart.control(cp = 0.01))
rpart.plot(mytree)
summary(mytree)

# To find optinal cp value
opt_cp <- mytree$cptable[which.min(mytree$cptable[, 'xerror']),'CP']
mytree2 <- rpart(win~kda + totdmgtochamp + totdmgtaken + champlvl + 
                   towerkills + inhibkills + duration + firstinhib + 
                   baronkills, data = lol, control = rpart.control(cp = opt_cp))
printcp(mytree2)
rpart.plot(mytree2)

# Test for accuracy
sample <- sample.split(lol$win, SplitRatio = 0.75)
train <- subset(lol,sample ==TRUE)
test <- subset(lol,sample == FALSE)
mytree2 <- rpart(win~kda + totdmgtochamp + totdmgtaken + champlvl + 
                   towerkills + inhibkills + duration + firstinhib + 
                   baronkills, data = train, control = rpart.control(cp = opt_cp))
pred_result <- predict(mytree2, test)

result <- ifelse(pred_result > 0.5, 1, 0)
accuracy <- mean(result == test$win)
accuracy
accuracy_score <- c(accuracy_score, accuracy)

###### Random Forest ######
library(randomForest)
myforest <- randomForest(win~kda + totdmgtochamp + totdmgtaken + champlvl + 
                           towerkills + inhibkills + duration + firstinhib + 
                           baronkills, ntree = 500, data = lol, importance = TRUE, na.action = na.omit)
myforest
importance(myforest)
varImpPlot(myforest)

# To get MSE
predicted_score <- predict(myforest, newdata = lol, ntree = 500)
mean((predicted_score - win) ^ 2)

# Out-of-bag Cross Validation
myforest_oob <- randomForest(win~kda + totdmgtochamp + totdmgtaken + champlvl + 
                               towerkills + inhibkills + duration + firstinhib + 
                               baronkills, ntree = 500, data = lol, importance = TRUE, na.action = na.omit, do.trace = 50)

# Test for accuracy
sample <- sample.split(lol$win, SplitRatio = 0.75)
train <- subset(lol,sample ==TRUE)
test <- subset(lol,sample == FALSE)
myforest <- randomForest(win~kda + totdmgtochamp + totdmgtaken + champlvl + 
                           towerkills + inhibkills + duration + firstinhib + 
                           baronkills, ntree = 500, data = train, importance = TRUE, na.action = na.omit)
pred_result <- predict(myforest, test)
result <- ifelse(pred_result > 0.5, 1, 0)
accuracy <- mean(result == test$win)
accuracy
accuracy_score <- c(accuracy_score, accuracy)

###### Boosted forest ######
library(gbm)
set.seed(100)
boosted <- gbm(win~kda + totdmgtochamp + totdmgtaken + champlvl + 
                 towerkills + inhibkills + duration + firstinhib + 
                 baronkills, data = lol, distribution = "bernoulli", n.trees = 10000, interaction.depth = 8)
summary(boosted)


# To get MSE
predicted_score <- predict(boosted, newdata = lol, n.trees = 10000)
mean((predicted_score - lol$win) ^ 2)

# Test for accuracy
sample <- sample.split(lol$win, SplitRatio = 0.75)
train <- subset(lol,sample ==TRUE)
test <- subset(lol,sample == FALSE)
boosted <- gbm(win~kda + totdmgtochamp + totdmgtaken + champlvl + 
                 towerkills + inhibkills + duration + firstinhib + 
                 baronkills, data = train, distribution = "bernoulli", n.trees = 10000, interaction.depth = 8)
pred_result <- predict(boosted, test)
result <- ifelse(pred_result > 0.5, 1, 0)
accuracy <- mean(result == test$win)
accuracy
accuracy_score <- c(accuracy_score, accuracy)

# model with all predictors, just for comparison
sample <- sample.split(lol$win, SplitRatio = 0.75)
train <- subset(lol,sample ==TRUE)
test <- subset(lol,sample == FALSE)
boosted <- gbm(win~., data = train, distribution = "bernoulli", n.trees = 10000, interaction.depth = 8)
pred_result <- predict(boosted, test)
result <- ifelse(pred_result > 0.5, 1, 0)
accuracy <- mean(result == test$win)
accuracy

# print the table of accuracy score of each model
acc <- data.frame(models, accuracy_score)
acc

######################################################################################
# Step 5: Clustering
######################################################################################
matches = read.csv(file = 'final_match_normalized.csv')
attach(matches)

matches$win <- ifelse(win == 0, 0, ifelse(win == 0.4, 0, ifelse(win == 0.6, 1, 1)))
table(matches$win)
dat = subset(matches, win == 1)
cluster_Dat = dat[,c(3:23)]

## detect outliers for clustering
## kda
lower_bound = quantile(cluster_Dat$kda, 0.001)
upper_bound = quantile(cluster_Dat$kda, 0.999)
outlier_ind_1 = which(cluster_Dat$kda < lower_bound | cluster_Dat$kda > upper_bound)

## totdmgdealt
lower_bound = quantile(cluster_Dat$totdmgdealt, 0.001)
upper_bound = quantile(cluster_Dat$totdmgdealt, 0.999)
outlier_ind_2 = which(cluster_Dat$totdmgdealt < lower_bound | cluster_Dat$totdmgdealt > upper_bound)

## totdmgtochamp
lower_bound = quantile(cluster_Dat$totdmgtochamp, 0.001)
upper_bound = quantile(cluster_Dat$totdmgtochamp, 0.999)
outlier_ind_3 = which(cluster_Dat$totdmgtochamp < lower_bound | cluster_Dat$totdmgtochamp > upper_bound)

## totdmgtaken
lower_bound = quantile(cluster_Dat$totdmgtaken, 0.001)
upper_bound = quantile(cluster_Dat$totdmgtaken, 0.999)
outlier_ind_4 = which(cluster_Dat$totdmgtaken < lower_bound | cluster_Dat$totdmgtaken > upper_bound)

## goldearned
lower_bound = quantile(cluster_Dat$goldearned, 0.001)
upper_bound = quantile(cluster_Dat$goldearned, 0.999)
outlier_ind_5 = which(cluster_Dat$goldearned < lower_bound | cluster_Dat$goldearned > upper_bound)

## goldspent
lower_bound = quantile(cluster_Dat$goldspent, 0.001)
upper_bound = quantile(cluster_Dat$goldspent, 0.999)
outlier_ind_6 = which(cluster_Dat$goldspent < lower_bound | cluster_Dat$goldspent > upper_bound)

## totminionskilled
lower_bound = quantile(cluster_Dat$totminionskilled, 0.001)
upper_bound = quantile(cluster_Dat$totminionskilled, 0.999)
outlier_ind_7 = which(cluster_Dat$totminionskilled < lower_bound | cluster_Dat$totminionskilled > upper_bound)

## neutralminionskilled
lower_bound = quantile(cluster_Dat$neutralminionskilled, 0.001)
upper_bound = quantile(cluster_Dat$neutralminionskilled, 0.999)
outlier_ind_8 = which(cluster_Dat$neutralminionskilled < lower_bound | cluster_Dat$neutralminionskilled > upper_bound)

## champlvl
lower_bound = quantile(cluster_Dat$champlvl, 0.001)
upper_bound = quantile(cluster_Dat$champlvl, 0.999)
outlier_ind_9 = which(cluster_Dat$champlvl < lower_bound | cluster_Dat$champlvl > upper_bound)

## towerkills
lower_bound = quantile(cluster_Dat$towerkills, 0.001)
upper_bound = quantile(cluster_Dat$towerkills, 0.999)
outlier_ind_10 = which(cluster_Dat$towerkills < lower_bound | cluster_Dat$towerkills > upper_bound)

## inhibkills
lower_bound = quantile(cluster_Dat$inhibkills, 0.001)
upper_bound = quantile(cluster_Dat$inhibkills, 0.999)
outlier_ind_11 = which(cluster_Dat$inhibkills < lower_bound | cluster_Dat$inhibkills > upper_bound)

## baronkills
lower_bound = quantile(cluster_Dat$baronkills, 0.001)
upper_bound = quantile(cluster_Dat$baronkills, 0.999)
outlier_ind_12 = which(cluster_Dat$baronkills < lower_bound | cluster_Dat$baronkills > upper_bound)

## dragonkills
lower_bound = quantile(cluster_Dat$dragonkills, 0.001)
upper_bound = quantile(cluster_Dat$dragonkills, 0.999)
outlier_ind_13 = which(cluster_Dat$dragonkills < lower_bound | cluster_Dat$dragonkills > upper_bound)

## dragonkills
lower_bound = quantile(cluster_Dat$harrykills, 0.001)
upper_bound = quantile(cluster_Dat$harrykills, 0.999)
outlier_ind_14 = which(cluster_Dat$harrykills < lower_bound | cluster_Dat$harrykills > upper_bound)

## store in dataframe
result = c(outlier_ind_1,outlier_ind_2,outlier_ind_3,outlier_ind_4,outlier_ind_5,
           outlier_ind_6,outlier_ind_7,outlier_ind_8,outlier_ind_9,outlier_ind_10,
           outlier_ind_11,outlier_ind_12,outlier_ind_13,outlier_ind_14)
sort(table(result))

## select frequency >=3
## final list of outliers
outlier_index = c(965,988833,892,936,1159)
cluster_Dat = cluster_Dat[-outlier_index,]

## Find cluster number
library(factoextra)
set.seed(123)
fviz_nbclust(cluster_Dat, kmeans, method = "silhouette") +
  labs(subtitle = "Silhouette method")

fviz_nbclust(cluster_Dat, kmeans, method = "wss")
set.seed(123)
km.4=kmeans(cluster_Dat,4) #4 clusters
df = aggregate(cluster_Dat, by=list(cluster=km.4$cluster), mean)


