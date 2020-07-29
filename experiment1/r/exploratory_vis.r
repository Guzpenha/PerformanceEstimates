library(ggplot2)
require(gridExtra)
library(dplyr)
### VISUALIZACAO 1 ###

df_errors = read.csv("/home/guzpenha/personal/msc-gustavo-penha/data/created/user_avg_errors.csv")
df_errors[0:2,]

means =  ddply(df_errors, .(RS),summarize, avg_error = mean(avg_error))
means[means$avg_error == min(means$avg_error),]

g1 <- ggplot(df_errors, aes(x=RS,y=avg_error)) + 
  geom_boxplot(notch = TRUE,outlier.shape = NA) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ylab("Average absolute error") + 
  xlab("Model")+
  ggtitle("Distribution of users average error by Recommender System") 
g1

df_errors_2 = df_errors[df_errors$RS!= "Constant",]
df_errors_2 = df_errors_2[df_errors_2$RS!= "Constant5",]
df_errors_2 = df_errors_2[df_errors_2$RS!= "Random",]
df_errors_2 = df_errors_2[df_errors_2$RS!= "GlobalAverage",]

g2 <- ggplot(df_errors_2, aes(x=RS,y=avg_error)) +  
  geom_boxplot(notch = TRUE,outlier.shape = NA) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ylab("Average absolute error") + 
  xlab("Model")+
  ggtitle("Distribution of users average error by Recommender System") 

g2

sampled_users = sample(18000, 10)
df_sampled = df_errors_2[df_errors_2$userId %in% sampled_users,]
df_sampled$userId = as.character(df_sampled$userId)

ggplot(df_sampled,aes(x = RS, y = avg_error))  +
  geom_path(aes(group=userId , color=userId))+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+ theme(legend.position="none") +
  ylab("Average absolute error") + 
  xlab("Model")

### VISUALIZACAO 2 ###

user_features_df = read.csv("/home/guzpenha/personal/msc-gustavo-penha/data/created/user_features.csv")
library(GGally)
library(ggplot2)
ggpairs(user_features_df[0:10,2:5])

# df_joined = merge(df_errors_2,user_features_df, by="userId")
df_joined = merge(df_errors,user_features_df, by="userId")

write.csv(df_joined, file = "/home/guzpenha/personal/msc-gustavo-penha/data/created/user_features_with_errors.csv")

df_joined$fwls_feature_4

g1 <- ggplot(df_joined[0:1000,],aes(x =fwls_feature_4 , y = avg_error)) + 
  geom_point() + facet_grid(.~ RS) + geom_smooth(method = lm)
g2 <- ggplot(df_joined[0:1000,],aes(x =fwls_feature_6 , y = avg_error)) + 
  geom_point() + facet_grid(.~ RS) + geom_smooth(method = lm)
g3 <- ggplot(df_joined[0:1000,],aes(x =fwls_feature16 , y = avg_error)) + 
  geom_point() + facet_grid(.~ RS) + geom_smooth(method = lm)
g4 <- ggplot(df_joined[0:1000,],aes(x =fwls_feature24 , y = avg_error)) + 
  geom_point() + facet_grid(.~ RS) + geom_smooth(method = lm)
 
grid.arrange(g1,g2,g3,g4,ncol = 1)

### VISUALIZACAO 3 ###

user2d_df = read.csv("/home/guzpenha/personal/msc-gustavo-penha/data/created/user_2d.csv")
user2d_df = user2d_df[(user2d_df$RS== "Constant5")  | (user2d_df$RS== "UserAverage") | (user2d_df$RS== "BiasedMatrixFactorization") ,]

# user2d_df = user2d_df[user2d_df$RS!= "Constant",]
# user2d_df = user2d_df[user2d_df$RS!= "Constant5",]
# user2d_df = user2d_df[user2d_df$RS!= "Random",]
# user2d_df = user2d_df[user2d_df$RS!= "GlobalAverage",]
user2d_df$label = as.character(user2d_df$label)
ggplot(user2d_df,aes(x=TSNE_0,y=TSNE_1,color=label)) + geom_point(alpha = 0.6) + facet_grid(. ~ RS) + scale_color_manual(values= c("#8cbdff","#ff23cf"))
