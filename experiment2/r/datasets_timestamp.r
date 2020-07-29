library(ggplot2)
require(gridExtra)
library(plyr)
library(zoo)
library(lubridate)

netflix = read.csv("/home/guz/ssd/msc-gustavo-penha/data/netflix/prepared_df.csv")
netflix$timestamp <- as.POSIXct(netflix$timestamp ,"%Y/%m/$d")
netflix$year <- year(netflix$timestamp)
netflix$month <- month(netflix$timestamp)

netflix_count <- count(netflix, c('year','month'))
netflix_count$Date <- with(netflix_count, sprintf("%d-%02d", year, month))

g1 <- ggplot(netflix_count,aes(x=Date,y=freq,group=1)) + geom_line()+ylab("Count of ratings") + 
  theme(axis.text.x=element_text(angle=45, hjust=1)) + ggtitle("netflix")
g1

ml100k = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/ml100k.csv")
ml100k$timestamp <- as.POSIXct(ml100k$timestamp,"%Y/%d/$m %H:%M:%OS")
ml100k$year <- year(ml100k$timestamp)
ml100k$month <- month(ml100k$timestamp)

ml100k_count <- count(ml100k, c('year','month'))
ml100k_count$Date <- with(ml100k_count, sprintf("%d-%02d", year, month))

g1 <- ggplot(ml100k_count,aes(x=Date,y=freq,group=1)) + geom_line()+ylab("Count of ratings") + 
  theme(axis.text.x=element_text(angle=45, hjust=1)) + ggtitle("ML100k")
g1

ml1m = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/ml1m.csv")
ml1m$timestamp <- as.POSIXct(ml1m$timestamp,"%Y/%d/$m %H:%M:%OS")
ml1m$year <- year(ml1m$timestamp)
ml1m$month <- month(ml1m$timestamp)

ml1m_count <- count(ml1m, c('year','month'))

ml1m_count$Date <- with(ml1m_count, sprintf("%d-%02d", year, month))
ml1m_count$data <- as.Date(as.yearmon(ml1m_count$Date))

g2 <- ggplot(ml1m_count,aes(x=Date,y=freq,group=1)) + 
  geom_line() + 
  ylab("Count of ratings") + 
  theme(axis.text.x=element_text(angle=45, hjust=1)) + ggtitle("ML1M")



ml20m = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/ml20m.csv")
ml20m$timestamp <- as.POSIXct(ml20m$timestamp,"%Y/%d/$m %H:%M:%OS")
ml20m[0:10,]
ml20m$year <- year(ml20m$timestamp)
ml20m$month <- month(ml20m$timestamp)

ml20m_count <- count(ml20m, c('year','month'))

ml20m_count$Date <- with(ml20m_count, sprintf("%d-%02d", year, month))
ml20m_count$data <- as.Date(as.yearmon(ml20m_count$Date))

g3 <- ggplot(ml20m_count,aes(x=Date,y=freq,group=1)) + 
  geom_line() + 
  ylab("Count of ratings") + 
  theme(axis.text.x=element_text(angle=70, hjust=1)) + ggtitle("ml20m")

grid.arrange(g1,g2,g3,ncol=1)

