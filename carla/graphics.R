rm(list=ls())
library(readr)
library(reshape2)
require(ggplot2)
library(dplyr)
library(stringr)
library(latex2exp)

#function required for legend generation
require(gridExtra)
g_legend <- function(a.gplot){
  tmp<- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

# if export legend is True, then, a legend is generated in a separated PDF file (and figure does not have a proper legend)
export_legend=T

######################################################################"

results <- read_csv("recourse_invalidation_results/formated_results/Results_adult.csv", col_types = cols(...1 = col_skip()))
results$Dataset <- "Adult"

results2 <- read_csv("recourse_invalidation_results/formated_results/Results_compas.csv", col_types = cols(...1 = col_skip()))
results2$Dataset <- "Compas"
results<-rbind(results , results2)

results2 <- read_csv("recourse_invalidation_results/formated_results/Results_give_me_some_credit.csv", col_types = cols(...1 = col_skip()))
results2$Dataset <- "GSC"
results<-rbind(results , results2)

rm(results2)

###### Invalidation wrt Distance: curves (mean) ######

appender <- function(string) 
  TeX(paste("$\\sigma^2 = $", string))  

data = results %>% mutate(Method = str_replace(Method, "wachter_rip", "PROBE")) %>% 
  mutate(Method = str_replace(Method, "robust_counterfactuals_random_v2", "CROCO")) %>% 
  mutate(Method = str_replace(Method, "wachter", "Wachter")) 

data = data %>% group_by(Target,Method,Sigma,Dataset) %>% summarise(
    Distance = mean(Distance),
    Estimator = mean(Estimator),
    Invalidation_rate = mean(Invalidation_rate))

p <- ggplot(data, mapping=aes(x=Distance, y=Invalidation_rate, color=Method, shape=Method))
p <- p + facet_grid(Sigma ~ Dataset, scales='free', labeller = labeller(Sigma=as_labeller(appender, default = label_parsed)))
#p <- p + geom_smooth()
p <- p + geom_point(size=2.)
p <- p + geom_path()
p <- p + xlab("Distance (L1)") + ggtitle("") + ylab(expression("Recourse invalidation rate ("~Gamma~")"))
p <- p + theme_bw()
p <- p + theme(strip.background = element_rect(colour="black", fill="white"), 
               legend.title=element_blank(),
               panel.grid.major = element_blank())

if(export_legend) {
  p <- p + theme(legend.position="none")
}
p

ggsave(filename="plot_R/curves.pdf", plot=p, width = 6, height = 6)

if(export_legend) {
  # Generate legend (to be saved manually)
  p <- p + theme(text = element_text(size=12),legend.position = "bottom") + guides(color=guide_legend(title=""),shape=guide_legend(title=""),linetype=guide_legend(title=""))
  legend <- g_legend(p)
  grid.arrange(legend)
  legend
  ggsave(filename="plot_R/legend_curves.pdf", plot=legend, width = 5.8, height = 0.4)
}
#######################################


###### Invalidation wrt Target: boxplots ######
data = results %>% mutate(Method = str_replace(Method, "wachter_rip", "PROBE")) %>% 
  mutate(Method = str_replace(Method, "robust_counterfactuals_random_v2", "CostERC")) %>% 
  mutate(Method = str_replace(Method, "wachter", "Wachter")) %>% filter(Method=="PROBE" | Method=="CostERC")

data$Target <- as.factor(data$Target)
data = data %>% na.omit()

p <- ggplot(data, mapping=aes(x=Target, y=Invalidation_rate, color=Method, shape=Method))
p <- p + facet_grid(Sigma ~ Dataset, scales='free', labeller = labeller(Sigma=as_labeller(appender, default = label_parsed)))
p <- p + geom_boxplot(outlier.shape = NA)
#p <- p + geom_boxplot()
#p <- p + geom_violin()
p <- p + xlab(TeX("Targeted recourse invalidation rate ($\\bar{\\Gamma}$)")) + ggtitle("") + ylab(TeX(" Estimated recourse invalidation rate ($\\Gamma$)"))
p <- p + theme_bw()
p <- p + theme(strip.background = element_rect(colour="black", fill="white"), 
               legend.title=element_blank(),
               panel.grid.major = element_blank())

if(export_legend) {
  p <- p + theme(legend.position="none")
}
p

ggsave(filename="plot_R/boxplots.pdf", plot=p, width = 6, height = 8)

if(export_legend) {
  # Generate legend (to be saved manually)
  p <- p + theme(text = element_text(size=12),legend.position = "bottom") + guides(color=guide_legend(title=""),shape=guide_legend(title=""),linetype=guide_legend(title=""))
  legend <- g_legend(p)
  grid.arrange(legend)
  legend
  ggsave(filename="plot_R/legend_boxplots.pdf", plot=legend, width = 5.8, height = 0.4)
}
#######################################


###### Invalidation wrt Target: boxplots ######
data = results %>% mutate(Method = str_replace(Method, "wachter_rip", "PROBE")) %>% 
  mutate(Method = str_replace(Method, "robust_counterfactuals_random_v2", "CostERC")) %>% 
  mutate(Method = str_replace(Method, "wachter", "Wachter")) %>% filter(Method=="PROBE" | Method=="CostERC") %>% filter(Dataset!="Adult")

data$Sigma <- as.factor(data$Sigma)
data = data %>% na.omit()

p <- ggplot(data, mapping=aes(x=Sigma, y=Invalidation_rate, color=Method, shape=Method))
p <- p + facet_grid(Target ~ Dataset, scales='free', labeller = labeller(Sigma=as_labeller(appender, default = label_parsed)))
p <- p + geom_boxplot(outlier.shape = NA)
p <- p + xlab(TeX("Targeted recourse invalidation rate ($\\bar{\\Gamma}$)")) + ggtitle("") + ylab(TeX(" Estimated recourse invalidation rate ($\\Gamma$)"))
p <- p + theme_bw()
p <- p + theme(strip.background = element_rect(colour="black", fill="white"), 
               legend.title=element_blank(),
               panel.grid.major = element_blank())

if(export_legend) {
  p <- p + theme(legend.position="none")
}
p

#######################################

###### Invalidation wrt Target: diag-plot with means ######
data = results %>% mutate(Method = str_replace(Method, "wachter_rip", "PROBE")) %>% 
  mutate(Method = str_replace(Method, "robust_counterfactuals_random_v2", "CostERC")) %>% 
  mutate(Method = str_replace(Method, "wachter", "Wachter")) %>% filter(Method=="PROBE" | Method=="CostERC")

data = data %>% group_by(Target,Method,Sigma,Dataset) %>% summarise(
  Distance = mean(Distance),
  Estimator = mean(Estimator),
  Invalidation_rate = mean(Invalidation_rate))

data = data %>% na.omit()

p <- ggplot(data, mapping=aes(x=Target, y=Invalidation_rate, color=Method, shape=Method))
p <- p + facet_grid(Sigma ~ Dataset, labeller = labeller(Sigma=as_labeller(appender, default = label_parsed)))
p <- p + geom_abline(intercept = 0, slope = 1 )
p <- p + coord_fixed(ratio=1)
p <- p + ylim(0,0.75) + xlim(0,0.75)
p <- p + geom_point()
p <- p + xlab(TeX("Targeted recourse invalidation rate ($\\bar{\\Gamma}$)")) + ggtitle("") + ylab(TeX(" Estimated invalidation rate ($\\Gamma$)"))
p <- p + theme_bw()
p <- p + theme( strip.background = element_rect(colour="black", fill="white"), 
                legend.title=element_blank(),
  panel.grid.major = element_blank())

if(export_legend) {
  p <- p + theme(legend.position="none")
}
p

ggsave(filename="plot_R/diagonale_mean.pdf", plot=p, width = 6, height = 8)

############################


###### Invalidation wrt Target: diag-plot all points ######
data = results %>% mutate(Method = str_replace(Method, "wachter_rip", "PROBE")) %>% 
  mutate(Method = str_replace(Method, "robust_counterfactuals_random_v2", "CROCO")) %>% 
  mutate(Method = str_replace(Method, "wachter", "Wachter")) %>% filter(Method=="PROBE" | Method=="CROCO")

data = data %>% na.omit()

p <- ggplot(data, mapping=aes(x=Target, y=Invalidation_rate, color=Method, shape=Method))
p <- p + facet_grid(Sigma ~ Dataset, labeller = labeller(Sigma=as_labeller(appender, default = label_parsed)))
p <- p + geom_abline(intercept = 0, slope = 1 )
p <- p + coord_fixed(ratio=1)
p <- p + ylim(0,0.75) + xlim(0,0.75)
p <- p + geom_point()
p <- p + xlab(TeX("Targeted recourse invalidation rate ($\\bar{\\Gamma}$)")) + ggtitle("") + ylab(TeX("Recourse invalidation rate ($\\Gamma$)"))
p <- p + theme_bw()
p <- p + theme( strip.background = element_rect(colour="black", fill="white"), 
                legend.title=element_blank(),
                panel.grid.major = element_blank())

if(export_legend) {
  p <- p + theme(legend.position="none")
}
p

ggsave(filename="plot_R/diagonale_all.pdf", plot=p, width = 6, height = 7)



if(export_legend) {
  # Generate legend (to be saved manually)
  p <- p + theme(text = element_text(size=12),legend.position = "bottom") + guides(color=guide_legend(title=""),shape=guide_legend(title=""),linetype=guide_legend(title=""))
  legend <- g_legend(p)
  grid.arrange(legend)
  legend
  ggsave(filename="plot_R/legend_diag.pdf", plot=legend, width = 5.8, height = 0.4)
}


###########################################

###### Invalidation wrt Distance: curves (std) ###### 

data = results %>% mutate(Method = str_replace(Method, "wachter_rip", "PROBE")) %>% 
  mutate(Method = str_replace(Method, "robust_counterfactuals_random_v2", "CostERC")) %>% 
  mutate(Method = str_replace(Method, "wachter", "Wachter")) 

data = data %>% group_by(Target,Method,Sigma,Dataset) %>% summarise(
    Distance = sd(Distance),
    Estimator = sd(Estimator),
    Invalidation_rate = sd(Invalidation_rate))

p <- ggplot(data, mapping=aes(x=Distance, y=Invalidation_rate, color=Method, shape=Method))
p <- p + facet_grid(Sigma ~ Dataset, scales='free', labeller = labeller(Sigma=as_labeller(appender, default = label_parsed)))
#p <- p + geom_smooth()
p <- p + geom_point(size=2.)
p <- p + geom_path()
p <- p + xlab("Distance (L1)") + ggtitle("") + ylab(expression("Recourse invalidation rate ("~Gamma~")"))
p <- p + theme_bw()
p <- p + theme(strip.background = element_rect(colour="black", fill="white"), 
               legend.title=element_blank(),
               panel.grid.major = element_blank())

if(export_legend) {
  p <- p + theme(legend.position="none")
}
p

ggsave(filename="plot_R/std_curves.pdf", plot=p, width = 6, height = 6)

if(export_legend) {
  # Generate legend (to be saved manually)
  p <- p + theme(text = element_text(size=12),legend.position = "bottom") + guides(color=guide_legend(title=""),shape=guide_legend(title=""),linetype=guide_legend(title=""))
  legend <- g_legend(p)
  grid.arrange(legend)
  legend
  ggsave(filename="plot_R/legend_curves_std.pdf", plot=legend, width = 5.8, height = 0.4)
}


###### Upper-bound vs recourse invalidation rate: diag-plot all points ######
data = results %>% mutate(Method = str_replace(Method, "wachter_rip", "PROBE")) %>% 
  mutate(Method = str_replace(Method, "robust_counterfactuals_random_v2", "CostERC")) %>% 
  mutate(Method = str_replace(Method, "wachter", "Wachter")) %>% filter(Method=="PROBE" | Method=="CostERC")

data = data %>% na.omit()

data <- filter(data,Method=="CostERC")

p <- ggplot(data, mapping=aes(x=2*(Estimator+0.1), y=Invalidation_rate,color=Method))
p <- p + facet_grid(Sigma ~ Dataset, labeller = labeller(Sigma=as_labeller(appender, default = label_parsed)))
p <- p + geom_abline(intercept = 0, slope = 1 )
p <- p + coord_fixed(ratio=1)
p <- p + ylim(0,1) + xlim(0,1)
p <- p + geom_point()
p <- p + xlab(expression("Upper bound value for CostERC ("*frac(m+tilde(Theta), 1-t)*")")) + ggtitle("") + ylab(expression(" Recourse invalidation rate ("*Gamma*")"))
#p <- p + xlab(TeX("Upper bound value for CostERC ($\\frac{m+\\tilde{\\Theta}}{1-t}$)") + ggtitle("") + ylab(TeX("Estimated recourse invalidation rate ($\\Gamma$)"))
p <- p + theme_bw()
p <- p + theme( strip.background = element_rect(colour="Black", fill="white"), 
                legend.title=element_blank(),
                panel.grid.major = element_blank(),panel.spacing.x = unit(4, "mm"))

if(export_legend) {
  p <- p + theme(legend.position="none")
}
p

ggsave(filename="plot_R/diagonale_upper_bound.pdf", plot=p, width = 6, height = 7)

if(export_legend) {
  # Generate legend (to be saved manually)
  p <- p + theme(text = element_text(size=12),legend.position = "bottom") + guides(color=guide_legend(title=""),shape=guide_legend(title=""),linetype=guide_legend(title=""))
  legend <- g_legend(p)
  grid.arrange(legend)
  legend
  ggsave(filename="plot_R/legend_diag_upper_bound.pdf", plot=legend, width = 5.8, height = 0.4)
}















