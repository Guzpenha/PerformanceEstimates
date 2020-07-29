library(ggplot2)
library(gridExtra)

df = read.csv("/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/dim_reduction.csv")


#Labels distribution
#png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/labels_dist.png",width=1000,height=1000)
g1<- ggplot(df[df$feature=="Error-Features trainset",],aes(label)) + geom_bar() + facet_grid(. ~dataset ,scales = "free") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                   size = 9, hjust = 1)) + coord_flip()
print(g1)
#dev.off()

#png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/dim_reduction.png",width=1000,height=1000)
#dimensionality reduction
g2<- ggplot(df, aes(x= TSNE_0, y=TSNE_1, color=label))  + geom_point(alpha=0.2,size=4) + facet_wrap(dataset  ~ feature, scales = "free")
print(g2)
#dev.off()
# ggplot(df, aes(x= TSNE_0, y=TSNE_1, shape=label))  + geom_point(alpha=0.5,size=4) + facet_grid(dataset ~ feature)

