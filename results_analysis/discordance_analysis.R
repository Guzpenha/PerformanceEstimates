library(ggplot2)
library(gridExtra)
library(reshape2)

df = read.csv("/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/discordance_data.csv")

# df$numeric_discordance <- as.numeric(df$numeric_discordance)

# g1<- ggplot(df) + 
#   stat_ecdf(aes(numeric_discordance),geom="step") +
#   facet_grid(.~dataset,scales="free") + ggtitle("Numeric discordance between base RS")

png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/discordance_analysis_ecdf.png",width=1000,height=1000)
g2<-ggplot(df) + 
  stat_ecdf(aes(list_discordance),geom="step") +
  facet_grid(.~dataset,scales="free") + ggtitle("List discordance between base RS")
print(g2)
dev.off()

# df$userId <- as.character(df$userId)
# df.m <-  melt(df, id.vars=c('userId','dataset','model'))
# ggplot(df.m,aes(x=userId,y=value)) +   
#   geom_bar(aes(fill = variable), position = "dodge", stat="identity")+facet_grid(.~dataset,scales="free")
# 

library(plyr)
# a<- ddply(df, c("model","dataset"), summarise, corr=cor(numeric_discordance, NDCG))
# a$metric <-"NDCG@20"
# a$metric_type <- "List performance"
# b<- ddply(df, c("model","dataset"), summarise, corr=cor(numeric_discordance, MAP))
# b$metric <-"MAP"
# b$metric_type <- "List performance"
# c <- ddply(df, c("model","dataset"), summarise, corr=cor(numeric_discordance, Precision.5))
# c$metric <-"Precision@05"
# c$metric_type <- "List performance"
# d <- ddply(df, c("model","dataset"), summarise, corr=cor(numeric_discordance, Precision.10))
# d$metric <-"Precision@10"
# d$metric_type <- "List performance"
# e <- ddply(df, c("model","dataset"), summarise, corr=cor(numeric_discordance, Precision.20))
# e$metric <-"Precision@20"
# e$metric_type <- "List performance"
# f <- ddply(df, c("model","dataset"), summarise, corr=cor(numeric_discordance, RR))
# f$metric <-"MRR"
# f$metric_type <- "List performance"
# g <- ddply(df, c("model","dataset"), summarise, corr=cor(numeric_discordance, MSE))
# g$metric <-"MSE"
# g$metric_type <- "Error"
# h <- ddply(df, c("model","dataset"), summarise, corr=cor(numeric_discordance, RMSE))
# h$metric <-"RMSE"
# h$metric_type <- "Error"
# i <- ddply(df, c("model","dataset"), summarise, corr=cor(numeric_discordance, MAE))
# i$metric <-"MAE"
# i$metric_type <- "Error"

# df_num_dis<-rbind.fill(a,b,c,d,e,f,g,h,i)

# g1<- ggplot(df_num_dis[df_num_dis$model!="0.0",],aes(x=model,y=corr)) +
#   geom_bar(aes(fill = metric), position = "dodge", stat="identity")+facet_grid(metric_type~dataset) + coord_flip() + ggtitle("Numeric discordance correlation")+
#   theme(axis.text.x = element_text(angle = 45, vjust = 1,
#                                    size = 9, hjust = 1))


a2 <- ddply(df, c("model","dataset"), summarise, corr=cor(list_discordance, NDCG))
a2$metric <-"NDCG@20"
b2 <- ddply(df, c("model","dataset"), summarise, corr=cor(list_discordance, MAP))
b2$metric <-"MAP"
c2 <- ddply(df, c("model","dataset"), summarise, corr=cor(list_discordance, Precision.5))
c2$metric <-"Precision@05"
d2 <- ddply(df, c("model","dataset"), summarise, corr=cor(list_discordance, Precision.10))
d2$metric <-"Precision@10"
e2 <- ddply(df, c("model","dataset"), summarise, corr=cor(list_discordance, Precision.20))
e2$metric <-"Precision@20"
f2 <- ddply(df, c("model","dataset"), summarise, corr=cor(list_discordance, RR))
f2$metric <-"MRR"
g2 <- ddply(df, c("model","dataset"), summarise, corr=cor(list_discordance, MSE))
g2$metric <-"MSE"
h2 <- ddply(df, c("model","dataset"), summarise, corr=cor(list_discordance, RMSE))
h2$metric <-"RMSE"
i2 <- ddply(df, c("model","dataset"), summarise, corr=cor(list_discordance, MAE))
i2$metric <-"MAE"

a2$metric_type <- "List performance"
b2$metric_type <- "List performance"
c2$metric_type <- "List performance"
d2$metric_type <- "List performance"
e2$metric_type <- "List performance"
f2$metric_type <- "List performance"
g2$metric_type <- "Error"
h2$metric_type <- "Error"
i2$metric_type <- "Error"

df2<-rbind.fill(a2,b2,c2,d2,e2,f2,g2,h2,i2)
df2[is.na(df2)] <- 0

png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/discordance_analysis_cor_to_performance.png",width=1000,height=1000)
g2<- ggplot(df2[(df2$model!="0.0") & (df2$metric != "0"),],aes(x=model,y=corr)) +
  geom_bar(aes(fill = metric), position = "dodge", stat="identity")+facet_grid(metric_type~dataset) + coord_flip() + ggtitle("List discordance correlation")+
  theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                   size = 9, hjust = 1))
print(g2)
dev.off()
# grid.arrange(g1,g2,ncol=2)

# df$userId <- NULL
# df$dataset <- NULL
# df$model <- NULL
# # 
# data.frame(cor(df$numeric_discordance,df[, !names(df) %in% c("numeric_discordance", "list_discordance")]))
# data.frame(cor(df$list_discordance,df[, !names(df) %in% c("numeric_discordance", "list_discordance")]))
# 
# # 
# # ggplot(melt(cor_df),aes(x=reorder(variable,value),y=value)) + geom_bar(stat="identity") +coord_flip()
# # 
