library(ggplot2)
library(gridExtra)
library(reshape2)
library(psych)
library(plyr)
library(zoo)
library(lubridate)

df = read.csv("/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/labels_data.csv")

df$rating_y = as.character(df$rating_y)
df$relevance = as.character(df$relevance)

g1<- ggplot(df[!is.na(df$rating_y),],aes(x=rating_y)) + geom_bar() + facet_grid(.~dataset, scales="free") + coord_flip()

g2<- ggplot(df[df$relevance != 0,],aes(x=relevance)) + geom_bar() + facet_grid(.~dataset, scales="free")+ coord_flip()

grid.arrange(g1,g2,ncol=1)


df[0:5,]


counts <- count(df,c("userId","dataset","relevance"))

counts[0:5,]
ggplot(counts[counts$relevance!= 0,],aes(x=freq,color=relevance)) +stat_ecdf()+ facet_grid(.~dataset,scales="free")
