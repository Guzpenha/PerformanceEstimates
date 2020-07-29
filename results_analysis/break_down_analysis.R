library(ggplot2)
library(gridExtra)


inc_df = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/incremental_analysis.csv")
# inc_df$f_index = as.character(inc_df$f_index)
ggplot(inc_df[inc_df$f_index<21,],aes(x=f_index,y=NDCG.20,fill=input_space,label=input_space,ymin=NDCG.20-ci, ymax=NDCG.20+ci)) +
  geom_bar(stat="identity", position=position_dodge(width=0.9), width = 0.7) +
  # geom_line(stat="identity") +
  xlab("number features in input space (incrementally added using Gini importance order)")+ 
  geom_errorbar(position=position_dodge(width=0.9), colour="black", width=.1) + ylab("NDCG@20") + 
  scale_fill_manual(values = c("#848484","#ff6f1c"), name="Input space") + 
  geom_text(aes(y = 0.35), size=2.5, position=position_dodge(width=0.9), angle=270) +  theme(legend.position = c(0.95, 0.2))

#########################
# DIMENSION REDUCTION
#########################

theme_set(theme_gray(base_size = 18))
df = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/tmp/reduced_eval_yelp.csv")
df$avg.ndcg <- df$X0

png(filename="/home/guz/ssd/msc-gustavo-penha/dim_reduction_amazon_yelp.png",width=977,height=660)
g1<-ggplot(df[df$meta.learner!="LinearReg",],aes(x=TSNE_0,y=TSNE_1,colour=meta.learner, shape=features, size=avg.ndcg))+ 
  geom_point() + scale_shape_manual(values = 0:18) + annotate(geom="text", x=10, y=9, label="pointwise", color="black",alpha=1, size=8)+ 
  annotate(geom="text", x=-5, y=-9, label="pairwise & listwise", color="black",alpha=1, size=8)
print(g1)
dev.off()

png(filename="/home/guz/ssd/msc-gustavo-penha/minimized_dim_reduction_amazon_yelp.png",width=250,height=250)
g1<-ggplot(df[df$meta.learner!="LinearReg",],aes(x=TSNE_0,y=TSNE_1,colour=meta.learner, shape=features, size=avg.ndcg))+ 
  geom_point() + scale_shape_manual(values = 0:18) + annotate(geom="text", x=10, y=9, label="pointwise", color="black",alpha=1, size=8)+ 
  annotate(geom="text", x=0, y=-9, label="pairwise & listwise", color="black",alpha=1, size=8)+ theme(axis.line=element_blank(),
                                                                                        axis.text.x=element_blank(),
                                                                                        axis.text.y=element_blank(),
                                                                                        axis.ticks=element_blank(),
                                                                                        axis.title.x=element_blank(),
                                                                                        axis.title.y=element_blank(),
                                                                                        legend.position="none")
print(g1)
dev.off()

df = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/tmp/reduced_eval_amazon_books.csv")
df$avg.ndcg <- df$X0
png(filename="/home/guz/ssd/msc-gustavo-penha/dim_reduction_amazon_books.png",width=977,height=660)
g1<-ggplot(df[df$meta.learner!="LinearReg",],aes(x=TSNE_0,y=TSNE_1,colour=meta.learner, shape=features, size=avg.ndcg))+ 
 geom_point() + scale_shape_manual(values = 0:18) + annotate(geom="text", x=15, y=12, label="pairwise & listwise", color="black",alpha=1, size=8)+ 
  annotate(geom="text", x=-15, y=-9, label="pointwise", color="black",alpha=1, size=8)
print(g1)
dev.off()

png(filename="/home/guz/ssd/msc-gustavo-penha/minimized_dim_reduction_amazon_books.png",width=250,height=250)
g1<-ggplot(df[df$meta.learner!="LinearReg",],aes(x=TSNE_0,y=TSNE_1,colour=meta.learner, shape=features, size=avg.ndcg))+ 
  geom_point() + scale_shape_manual(values = 0:18) + annotate(geom="text", x=5, y=12, label="pairwise & listwise", color="black",alpha=1, size=8)+ 
  annotate(geom="text", x=-12, y=-9, label="pointwise", color="black",alpha=1, size=8)+ theme(axis.line=element_blank(),
                                                                                        axis.text.x=element_blank(),
                                                                                        axis.text.y=element_blank(),
                                                                                        axis.ticks=element_blank(),
                                                                                        axis.title.x=element_blank(),
                                                                                        axis.title.y=element_blank(),
                                                                                        legend.position="none")
print(g1)
dev.off()
df = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/tmp/reduced_eval_amazon_movies.csv")
df$avg.ndcg <- df$X0
png(filename="/home/guz/ssd/msc-gustavo-penha/dim_reduction_amazon_movies.png",width=977,height=660)
g1<-ggplot(df[df$meta.learner!="LinearReg",],aes(x=TSNE_0,y=TSNE_1,colour=meta.learner, shape=features, size=avg.ndcg))+ 
  geom_point() + scale_shape_manual(values = 0:18) + annotate(geom="text", x=9, y=-9, label="pairwise & listwise", color="black",alpha=1, size=8)+ 
  annotate(geom="text", x=-8, y=12, label="pointwise", color="black",alpha=1, size=8)
print(g1)
dev.off()

png(filename="/home/guz/ssd/msc-gustavo-penha/minimized_dim_reduction_amazon_movies.png",width=250,height=250)
g1<-ggplot(df[df$meta.learner!="LinearReg",],aes(x=TSNE_0,y=TSNE_1,colour=meta.learner, shape=features, size=avg.ndcg))+ 
  geom_point() + scale_shape_manual(values = 0:18) + annotate(geom="text", x=3, y=-9, label="pairwise & listwise", color="black",alpha=1, size=8)+ 
  annotate(geom="text", x=-9, y=14, label="pointwise", color="black",alpha=1, size=8)+ theme(axis.line=element_blank(),
                                                                                        axis.text.x=element_blank(),
                                                                                        axis.text.y=element_blank(),
                                                                                        axis.ticks=element_blank(),
                                                                                        axis.title.x=element_blank(),
                                                                                        axis.title.y=element_blank(),
                                                                                        legend.position="none")
print(g1)
dev.off()
df = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/tmp/reduced_eval_amazon_electronics.csv")
df$avg.ndcg <- df$X0
png(filename="/home/guz/ssd/msc-gustavo-penha/dim_reduction_amazon_electronics.png",width=977,height=660)
g1<-ggplot(df[df$meta.learner!="LinearReg",],aes(x=TSNE_0,y=TSNE_1,colour=meta.learner, shape=features, size=avg.ndcg))+ 
  geom_point() + scale_shape_manual(values = 0:18) + annotate(geom="text", x=15, y=-9, label="pointwise", color="black",alpha=1, size=8)+ 
  annotate(geom="text", x=-15, y=12, label="pairwise & listwise", color="black",alpha=1, size=8)
print(g1)
dev.off()

png(filename="/home/guz/ssd/msc-gustavo-penha/minimized_dim_reduction_amazon_electronics.png",width=250,height=250)
g1<-ggplot(df[df$meta.learner!="LinearReg",],aes(x=TSNE_0,y=TSNE_1,colour=meta.learner, shape=features, size=avg.ndcg))+ 
  geom_point() + scale_shape_manual(values = 0:18) + annotate(geom="text", x=13, y=-9, label="pointwise", color="black",alpha=1, size=8)+ 
  annotate(geom="text", x=-7, y=12, label="pairwise & listwise", color="black",alpha=1, size=8) + theme(axis.line=element_blank(),
                                                                                         axis.text.x=element_blank(),
                                                                                         axis.text.y=element_blank(),
                                                                                         axis.ticks=element_blank(),
                                                                                         axis.title.x=element_blank(),
                                                                                         axis.title.y=element_blank(),
                                                                                         legend.position="none")
print(g1)
dev.off()
# df = read.csv("/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/user_ensemble_winners.csv")
# g1<- ggplot(df,aes(winner_LTRERS)) + geom_bar() + ylab("# win") + xlab("") + facet_grid(.~dataset)
# g2<- ggplot(df,aes(winner_STREAM)) + geom_bar() + ylab("# win") + xlab("") + facet_grid(.~dataset)
# # g3<- ggplot(df,aes(winner_FWLS)) + geom_bar() + ylab("# win") + xlab("") + facet_grid(.~dataset)
# g4<- ggplot(df,aes(winner_SCB)) + geom_bar() + ylab("# win") + xlab("Ensemble") + facet_grid(.~dataset)
# grid.arrange(g1,g2,g4,ncol=1)
# df2 = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/results/raw_eval.csv")
# ggplot(df2[df2$model != "",],aes(x=reorder(model,NDCG),y=NDCG)) + geom_boxplot(notch=TRUE) + facet_grid(.~dataset) + coord_flip() + xlab("Ensemble")

# df3 = read.csv("/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/users_raw_enhanced.csv")
# df3$userId = as.character(df3$userId)
# g3<- ggplot(df3[df3$group == "LTRERS",], aes(x=userId,y=NDCG,fill=model)) + 
#   geom_bar(position="dodge", stat="summary", fun.y="mean") + facet_grid(dataset ~ winner_LTRERS, scales = "free_x", space = "free_x")+ xlab("User") +
#   theme(axis.text.x = element_text(angle = 45, vjust = 1,
#                                    size = 9, hjust = 1))
# g3
# png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/per_user_break_down.png",width=3000,height=500)
# print(g3)
# dev.off()

df_h2 = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/delta_h2.csv")
df_h2$delta = as.numeric(df_h2$delta)
df_h2$users <- as.numeric(row.names(df_h2))

# g1<-ggplot(df_h2,aes(x=reorder(users,-delta),y=delta)) +
#   geom_bar(stat="identity")+ xlab("Users") + ylab("(e) - (a) NDCG@20 delta")+ 
#   theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) + ggtitle("H1")  + theme(panel.grid.major = element_blank())
# g1 

theme_set(theme_gray(base_size = 18))

df_h1e = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/delta_h1e.csv")
df_h1e$delta = as.numeric(df_h1e$delta)
df_h1e$users <- as.numeric(row.names(df_h1e))
g2<-ggplot(df_h1e,aes(x=reorder(users,-delta),y=delta)) +
  geom_bar(stat="identity",width=1)+ xlab("User") + ylab("(a) - (b) NDCG@20 delta")+
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) + ggtitle("Q5 (Table 6)")+ theme(panel.grid.major = element_blank())

g2
  
df_h1d = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/delta_h1d.csv")
df_h1d$delta = as.numeric(df_h1d$delta)
df_h1d$users <- as.numeric(row.names(df_h1d))
g3<-ggplot(df_h1d,aes(x=reorder(users,-delta),y=delta)) +
  geom_bar(stat="identity",width=1)+ xlab("User") + ylab("(a) - (b) NDCG@20 delta")+ 
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) + ggtitle("Q4 (Table 5)")+ theme(panel.grid.major = element_blank())

df_h1c = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/delta_h1c.csv")
df_h1c$delta = as.numeric(df_h1c$delta)
df_h1c$users <- as.numeric(row.names(df_h1c))
g4<-ggplot(df_h1c,aes(x=reorder(users,-delta),y=delta)) +
  geom_bar(stat="identity",width=1)+ xlab("User") + ylab("(a) - (b) NDCG@20 delta")+ 
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) + ggtitle("Q3")+ theme(panel.grid.major = element_blank())


df_h1b = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/delta_h1b.csv")
df_h1b$delta = as.numeric(df_h1b$delta)
df_h1b$users <- as.numeric(row.names(df_h1b))
g5<-ggplot(df_h1b,aes(x=reorder(users,-delta),y=delta)) +
  geom_bar(stat="identity",width=1)+ xlab("User") + ylab("(c) - (b) NDCG@20 delta")+ 
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) + ggtitle("Q1 (Table 4)")+ theme(panel.grid.major = element_blank())

df_h1a = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/delta_h1a.csv")
df_h1a$delta = as.numeric(df_h1a$delta)
df_h1a$users <- as.numeric(row.names(df_h1a))
g6<-ggplot(df_h1a,aes(x=reorder(users,-delta),y=delta)) +
  geom_bar(stat="identity",width=1)+ xlab("User") + ylab("(c) - (a) NDCG@20 delta")+ 
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) + ggtitle("Q1 (Table 4)")+ theme(panel.grid.major = element_blank())

# grid.arrange(grid.arrange(g6,g5,g4,g3,g2,ncol=5),
#              grid.arrange(g6,g5,g4,g3,g2,ncol=5),ncol=1)

grid.arrange(g6,g5,g4,g3,g2,ncol=5)


###########################
#MINIMIZED VERSION
###########################

df_h1e = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/delta_h1e.csv")
df_h1e$delta = as.numeric(df_h1e$delta)
df_h1e$users <- as.numeric(row.names(df_h1e))
g2<-ggplot(df_h1e,aes(x=reorder(users,-delta),y=delta)) +
  geom_bar(stat="identity",width=1)+ xlab("User") + ylab("(a) - (b) NDCG@20 delta")+
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) + ggtitle("Q5")+ theme(panel.grid.major = element_blank())+ theme(axis.line=element_blank(),
                                                                                                                                               axis.text.x=element_blank(),
                                                                                                                                               axis.text.y=element_blank(),
                                                                                                                                               axis.ticks=element_blank(),
                                                                                                                                               axis.title.x=element_blank(),
                                                                                                                                               axis.title.y=element_blank(),
                                                                                                                                               legend.position="none")
g2

df_h1d = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/delta_h1d.csv")
df_h1d$delta = as.numeric(df_h1d$delta)
df_h1d$users <- as.numeric(row.names(df_h1d))
g3<-ggplot(df_h1d,aes(x=reorder(users,-delta),y=delta)) +
  geom_bar(stat="identity",width=1)+ xlab("User") + ylab("(a) - (b) NDCG@20 delta")+ 
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) + ggtitle("Q4")+ theme(panel.grid.major = element_blank()) + theme(axis.line=element_blank(),
                                                                                                                                                axis.text.x=element_blank(),
                                                                                                                                                axis.text.y=element_blank(),
                                                                                                                                                axis.ticks=element_blank(),
                                                                                                                                                axis.title.x=element_blank(),
                                                                                                                                                axis.title.y=element_blank(),
                                                                                                                                                legend.position="none")
df_h1c = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/delta_h1c.csv")
df_h1c$delta = as.numeric(df_h1c$delta)
df_h1c$users <- as.numeric(row.names(df_h1c))
g4<-ggplot(df_h1c,aes(x=reorder(users,-delta),y=delta)) +
  geom_bar(stat="identity",width=1)+ xlab("User") + ylab("(a) - (b) NDCG@20 delta")+ 
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) + ggtitle("Q3")+ theme(panel.grid.major = element_blank())+ theme(axis.line=element_blank(),
                                                                                                                                     axis.text.x=element_blank(),
                                                                                                                                     axis.text.y=element_blank(),
                                                                                                                                     axis.ticks=element_blank(),
                                                                                                                                     axis.title.x=element_blank(),
                                                                                                                                     axis.title.y=element_blank(),
                                                                                                                                     legend.position="none")


df_h1b = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/delta_h1b.csv")
df_h1b$delta = as.numeric(df_h1b$delta)
df_h1b$users <- as.numeric(row.names(df_h1b))
g5<-ggplot(df_h1b,aes(x=reorder(users,-delta),y=delta)) +
  geom_bar(stat="identity",width=1)+ xlab("User") + ylab("(c) - (b) NDCG@20 delta")+ 
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) + ggtitle("Q1")+ theme(panel.grid.major = element_blank())+ theme(axis.line=element_blank(),
                                                                                                                                               axis.text.x=element_blank(),
                                                                                                                                               axis.text.y=element_blank(),
                                                                                                                                               axis.ticks=element_blank(),
                                                                                                                                               axis.title.x=element_blank(),
                                                                                                                                               axis.title.y=element_blank(),
                                                                                                                                               legend.position="none")
df_h1a = read.csv("/home/guz/ssd/msc-gustavo-penha/experiment2/created_data/delta_h1a.csv")
df_h1a$delta = as.numeric(df_h1a$delta)
df_h1a$users <- as.numeric(row.names(df_h1a))
g6<-ggplot(df_h1a,aes(x=reorder(users,-delta),y=delta)) +
  geom_bar(stat="identity",width=1)+ xlab("User") + ylab("(c) - (a) NDCG@20 delta")+ 
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) + ggtitle("Q1")+ theme(panel.grid.major = element_blank())+theme(axis.line=element_blank(),
                                                                                                                                                axis.text.x=element_blank(),
                                                                                                                                                axis.text.y=element_blank(),
                                                                                                                                                axis.ticks=element_blank(),
                                                                                                                                                axis.title.x=element_blank(),
                                                                                                                                                axis.title.y=element_blank(),
                                                                                                                                                legend.position="none")
# grid.arrange(grid.arrange(g6,g5,g4,g3,g2,ncol=5),
#              grid.arrange(g6,g5,g4,g3,g2,ncol=5),ncol=1)

grid.arrange(g6,g5,g4,g3,g2,ncol=5)


df4 = read.csv("/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/users_delta.csv")
library(dplyr)

# for (d in unique(df$dataset)){
d = "amazon_books"

df = df4[df4$dataset == d, ]
df = df[df$delta_EF.val_MF != 0,]
pd <- df %>%
  group_by(group) %>%
  ungroup() %>%
  arrange(group, - delta_EF.val_MF) %>%
  mutate(order = row_number())
g4<- ggplot(pd, aes(x=order,y=delta_EF.val_MF)) +
  geom_bar(position="dodge", stat="summary", fun.y="mean") + facet_grid(dataset ~ group, scales = "free_x", space = "free_x")+ xlab("User") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                   size = 9, hjust = 1)) + ylab("NDCG delta between EF-val and MF") +  labs(fill="")  + ggtitle("Better than MF")
# png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/per_user_break_down_deltas_against_MF.png",width=1500,height=500)
print(g4)
# dev.off()

df = df4[df4$dataset == d, ]
df = df[df$delta_EF.val_None != 0,]
df = df[!is.na(df$dataset),]
pd <- df %>%
  group_by(group) %>%
  ungroup() %>%
  arrange(group, -delta_EF.val_None) %>%
  mutate(order = row_number())
g7<- ggplot(pd, aes(x=order,y=delta_EF.val_None)) +
  geom_bar(position="dodge", stat="summary", fun.y="mean") + facet_grid(dataset ~ group, scales = "free_x", space = "free_x")+ xlab("User") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                   size = 9, hjust = 1)) + ylab("NDCG delta between EF-val and None") +  labs(fill="") + ggtitle("Better than not using any additional features")
# png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/per_user_break_down_deltas_against_EF_No_features.png",width=1500,height=500)
print(g7)
# dev.off()

df = df4[df4$dataset == d, ]
df = df[df$delta_All.MF != 0,]
df = df[!is.na(df$dataset),]
pd <- df %>%
  group_by(group) %>%
  ungroup() %>%
  arrange(group, -delta_All.MF) %>%
  mutate(order = row_number())
g8<- ggplot(pd, aes(x=order,y=delta_All.MF)) +
  geom_bar(position="dodge", stat="summary", fun.y="mean") + facet_grid(dataset ~ group, scales = "free_x", space = "free_x")+ xlab("User") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                   size = 9, hjust = 1)) + ylab("NDCG delta between All and MF") +  labs(fill="") +ggtitle("Complementary to using MF only")
# png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/per_user_break_down_deltas_against_EF_No_features.png",width=1500,height=500)
print(g8)
# dev.off()


df = df4[df4$dataset == d, ]
df = df[df$delta_All.MF != 0,]
pd <- df %>%
  group_by(group) %>%
  ungroup() %>%
  arrange(group, -delta_All.MF) %>%
  mutate(order = row_number())
plots <- list()
i <-0
for( dimension in c("abnormality","abnormalityCR","avgRatingValue","fwls_feature16","fwls_feature24","fwls_feature_4","fwls_feature_6","moviesAvgRatings","moviesPopularity","ratingStdDeviation","support")){
i<-i+1
p<- ggplot(pd, aes_string(x="order",y="delta_All.MF",fill = dimension)) + 
  geom_bar(position="dodge", stat="summary", fun.y="mean") + facet_grid(dataset ~ group, scales = "free_x", space = "free_x")+ xlab("User") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                   size = 9, hjust = 1)) + ylab("EF - MF")  +
  labs(fill="") + ggtitle(dimension) +
  theme(
    # axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())+ scale_fill_gradient(low="green", high="red")
plots[[i]] <- p
# print(p)
}

# png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/per_user_break_down_deltas_by_feature_0.png",width=3000,height=2000)
print(grid.arrange(plots[[1]],plots[[2]],plots[[3]],plots[[4]],plots[[5]],plots[[6]],plots[[7]],plots[[8]],plots[[9]], plots[[10]],plots[[11]], ncol=4))
# dev.off()

# png(filename="/home/guz/ssd/msc-gustavo-penha/results_analysis/created_data/img/per_user_break_down_deltas_by_feature_1.png",width=1000,height=500)
# dev.off()

# }
