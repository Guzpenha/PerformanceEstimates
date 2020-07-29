library(ggplot2)
library(gridExtra)
library(dplyr)

df = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/results/robustness_analysis.csv")

pd <- df %>%
group_by(model,features) %>%
summarise(mean = mean(NDCG))

max(pd$mean)

# total <- merge(df,pd,by=c("model","features"))
png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/robustness_to_model.png",width=1800,height=350)
g1 <- ggplot(df,aes(x=model ,y=NDCG, fill=features)) + geom_boxplot(notch=TRUE) + facet_grid(dataset ~ ensemble, scales="free") + xlab("Ensemble")
print(g1)
dev.off()