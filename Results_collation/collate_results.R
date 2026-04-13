# File to collate all output results into a single table
# From Helen

#library(list)
library(dplyr)
library(plyr)
library(ggplot2)
library(ggthemes)
#library(ggsci)
theme_set(theme_bw())

# First combine the output files for overall performance
filenames <- list.files("model_results/output", pattern = "*.csv", full.names = TRUE)
ldf <- lapply(filenames, read.csv)
#ldf <- mapply(cbind, ldf, "filename" = filenames, SIMPLIFY = FALSE)
output <- ldply(ldf, data.frame)


output$prop <- rowSums(output[c("true_pos", "false_ng")]) / rowSums(output[c("true_neg", "false_ng", "true_pos", "false_pos")])

write.csv(output, paste0("Results_collation/Result_tables/all_outputs.csv"))