library(tidyverse)

select_min_fre <- function(df, ...) {
  df_min <- df %>% 
    group_by(ID) %>% 
    filter(...) %>% 
    slice_min(order_by = fre, with_ties = FALSE) # Only keep the first minimum (multiple may be available)
  return(df_min)
}

# Process table of all matrix IDs
data_cond_coarse = read_csv("output_512_min_cond_coarse.csv")
data_static = read_csv("output_512_static.csv")
data_random = read_csv("output_512_random_32_100.csv")
data_rows_cond = read_csv("output_512_rows_cond.csv")
data_rows_det = read_csv("output_512_rows_det.csv")

# Select rows with minimal FRE
data_cond_coarse_min_fre <- select_min_fre(data_cond_coarse)
data_static_min_fre <- select_min_fre(data_static)
data_random_min_fre <- select_min_fre(data_random)
data_rows_cond_min_fre <- select_min_fre(data_rows_cond)
data_rows_det_min_fre <- select_min_fre(data_rows_det)

# Select partition of fixed size
data_static_M32 <- data_static %>% filter(M == 32)
  
# Create bar plot for different methods
# TODO normalize the data (with 1 == data_static_M32['fre'])
barplot_data <- matrix(nrow=6, ncol=20)

barplot_data[1,] <- data_static_M32[["fre"]]
barplot_data[2,] <- data_static_min_fre[["fre"]]
barplot_data[3,] <- data_cond_coarse_min_fre[["fre"]]
barplot_data[4,] <- data_random_min_fre[["fre"]]
barplot_data[5,] <- data_rows_cond_min_fre[["fre"]]
barplot_data[6,] <- data_rows_det_min_fre[["fre"]]

par(mfrow=c(4,5))
for (i in seq(1,20)) {
  title <- sprintf("Matrix %d", i)
  barplot(height=barplot_data[,i], col=c(1:4), log="y", main=title, legend.text=T)
}
legend("topright", legend=c("S", "H0", "H1", "H2"), fill=c(1:4))
