library(tidyverse)

select_min_fre <- function(df, label) {
  df_min <- df %>% group_by(ID) %>% 
    slice_min(order_by = fre, with_ties = FALSE) %>%
    add_column(label = label)
  return(df_min)
}

# TODO: make order_by variable
select_min_cond <- function(df, label) {
  df_min <- df %>% group_by(ID) %>% 
    slice_min(order_by = cond_coarse, with_ties = FALSE) %>% 
    add_column(label = label)
  return(df_min)
}

select_common <- function(df) {
  df_common <- df %>% 
    select(ID, N, fre, cond_coarse, label)
  return(df_common)
}

# Process table of all matrix IDs
data_cond_coarse = read_csv("output_512_min_cond_coarse.csv")
data_static = read_csv("output_512_static.csv")
data_random = read_csv("output_512_random_32_100.csv")
data_rows_cond = read_csv("output_512_rows_cond.csv")
data_rows_det = read_csv("output_512_rows_det.csv")

# Rows for partition of fixed size
data_static_M32 <- data_static %>% 
  filter(M == 32) %>% add_column(label = "static_32")

# Select rows with minimal FRE
data_static_min_fre <- select_min_fre(data_static, "static")
data_cond_coarse_min_fre <- select_min_fre(data_cond_coarse, "cond_coarse")
data_random_min_fre <- select_min_fre(data_random, "random_min_fre")
data_random_min_cond <- select_min_cond(data_random, "random_min_cond")
data_rows_cond_min_fre <- select_min_fre(data_rows_cond, "rows_cond")
data_rows_det_min_fre <- select_min_fre(data_rows_det, "rows_det")

# Combine these tables with an additional column for the method
data_min_fre <- select_common(data_static_M32) %>%
  full_join(select_common(data_static_min_fre)) %>% 
  full_join(select_common(data_cond_coarse_min_fre)) %>%
  full_join(select_common(data_random_min_fre)) %>%
  full_join(select_common(data_random_min_cond)) %>%
  full_join(select_common(data_rows_cond_min_fre)) %>%
  full_join(select_common(data_rows_det_min_fre))

write_csv(data_min_fre, "output_merged.csv")

# Create bar plot for different methods
# TODO normalize the data (with 1 == data_static_M32['fre']) -> add extra column with "rating"
barplot_data <- matrix(nrow=7, ncol=20)

barplot_data[1,] <- data_static_M32[["fre"]]
barplot_data[2,] <- data_static_min_fre[["fre"]]
barplot_data[3,] <- data_cond_coarse_min_fre[["fre"]]
barplot_data[4,] <- data_random_min_fre[["fre"]]
barplot_data[5,] <- data_random_min_cond[["fre"]]
barplot_data[6,] <- data_rows_cond_min_fre[["fre"]]
barplot_data[7,] <- data_rows_det_min_fre[["fre"]]

par(mfrow=c(4,5))
for (i in seq(1,20)) {
  title <- sprintf("Matrix %d", i)
  barplot(height=barplot_data[,i], col=c(1:4), log="y", main=title, legend.text=T)
}
#legend("topright", legend=c("S", "H0", "H1", "H2"), fill=c(1:4))
