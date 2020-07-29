library(ggplot2)
library(gridExtra)

df = read.csv("/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/clf_f_importance.csv")

# png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/f_importance_SCB.png",width=1000,height=2500)
for (d in unique(df$dataset)){
g1 <- ggplot(df[df$dataset == d,], aes(x= reorder(feature,importance), y=importance, fill=feature_type) ) + 
  geom_bar(stat = "identity") + coord_flip() + facet_grid(. ~ dataset,scales = "free")+
  ylab("feature importance") + xlab("Feature") +
  geom_errorbar(aes(ymin=importance - std, ymax=importance + std), width=.1) + ggtitle("SCB classifier")+
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold"))

  # ggsave(g1)
print(g1)
}

# dev.off()

df_reg = read.csv("/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/regr_f_importance.csv")
# png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/f_importance_regressor.png",width=1000,height=2500)
for (d in unique(df$dataset)){
g2 <- ggplot(df_reg[df_reg$dataset == d,], aes(x= reorder(feature,importance), y=importance, fill=feature_type) ) + 
  geom_bar(stat = "identity") + coord_flip() + facet_grid(. ~ dataset)+
  ylab("feature importance") + xlab("Feature") +
  geom_errorbar(aes(ymin=importance - std, ymax=importance + std), width=.1) + ggtitle("STREAM regressor")
  # ggsave(g2)
print(g2)
}
# dev.off()

df_ltr = read.csv("/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/ltr_f_importance.csv")
# png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/f_importance_l2r.png",width=1000,height=2500)
for (d in unique(df$dataset)){
g3 <-  ggplot(df_ltr[df_ltr$dataset == d,], aes(x= reorder(feature,importance), y=importance, fill=feature_type) ) + 
  geom_bar(stat = "identity") + coord_flip() + facet_grid(. ~ dataset)+
  ylab("feature importance") + xlab("Feature") +
  geom_errorbar(aes(ymin=importance - std, ymax=importance + std), width=.1) + ggtitle("LTRERS learning to rank")

  # ggsave(g3)
print(g3)
}

# dev.off()
