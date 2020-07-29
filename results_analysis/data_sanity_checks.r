library(ggplot2)
library(gridExtra)
library(reshape2)
library(psych)
library(plyr)
library(zoo)
library(lubridate)

divided_df = read.csv("/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/divided_data.csv")

# divided_df[0:10,]
nrow(divided_df)
divided_df <- divided_df[!is.na(divided_df$timestamp),]
nrow(divided_df)

divided_df$timestamp <- as.Date(divided_df$timestamp)
divided_df$year <- year(divided_df$timestamp)
divided_df$month <- month(divided_df$timestamp)

divided_df <- count(divided_df, c('year','month','dataset','set'))
divided_df$Date <- with(divided_df, sprintf("%d-%02d", year, month))

# divided_df[0:10,]

g1 <- ggplot(divided_df[divided_df$Date!="NA-NA",],aes(x=Date,y=freq,group=1,color = set)) + geom_line()+ylab("Count of ratings") + 
  theme(axis.text.x=element_text(angle=45, hjust=1)) +facet_grid(dataset~. , scales = "free") + 
  geom_point(size=0.5)+
  scale_x_discrete(breaks=c("1998-01", "2000-01", "2002-01","2004-01","2006-01","2008-01","2010-01","2012-01","2014-01","2016-01"))

g1

divided_df = read.csv("/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/divided_data.csv")
divided_df[0:5,]
counts_per_user <- count(divided_df, c("dataset","userId","set"))
counts_per_user[0:5,]

g2 <- ggplot(counts_per_user[counts_per_user$set %in% c("Train RS","Validation"),],aes(x=freq,color=set)) + stat_ecdf() + facet_grid(. ~ dataset,scales="free") + xlab("Count of ratings for the user")
g2

# install.packages('reshape')
# library(reshape)
plots <- list()
i <-0
for (dataset in unique(counts_per_user$dataset)){
  i<-i+1
  casted_df <- dcast(counts_per_user[counts_per_user$dataset == dataset,], userId ~ set, sum)
  print(casted_df[0:5,])
  casted_df$delta <- casted_df$`Train RS`-casted_df$Validation
  plots[[i]]<- (ggplot(casted_df,aes(x="users",y=delta))+geom_boxplot()+ xlab("")+ylab("Delta train_count_ratings - val_count_ratings ")+ggtitle(dataset) + coord_flip()) 
}
print(grid.arrange(plots[[1]],plots[[2]],plots[[3]],plots[[4]], ncol=1))



