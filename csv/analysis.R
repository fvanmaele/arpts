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
data_static_min_fre <- select_min_fre(data_static, "static_min_fre")
data_static_min_cond <- select_min_cond(data_static, "static_min_cond")
data_cond_coarse_min_fre <- select_min_fre(data_cond_coarse, "cond_coarse_min_fre")
data_cond_coarse_min_cond <- select_min_cond(data_cond_coarse, "cond_coarse_min_cond")
data_rows_cond_min_fre <- select_min_fre(data_rows_cond, "rows_cond_min_fre")
data_rows_cond_min_cond <- select_min_cond(data_rows_cond, "rows_cond_min_cond")
data_rows_det_min_fre <- select_min_fre(data_rows_det, "rows_det_min_fre")
data_rows_det_min_cond <- select_min_cond(data_rows_det, "rows_det_min_cond")
data_random_min_fre <- select_min_fre(data_random, "random_min_fre")
data_random_min_cond <- select_min_cond(data_random, "random_min_cond")

# Combine these tables with an additional column for the method
data_mins <- select_common(data_static_M32) %>%
  full_join(select_common(data_static_min_fre)) %>% 
  full_join(select_common(data_static_min_cond)) %>% 
  full_join(select_common(data_cond_coarse_min_fre)) %>%
  full_join(select_common(data_cond_coarse_min_cond)) %>%
  full_join(select_common(data_rows_cond_min_fre)) %>%
  full_join(select_common(data_rows_cond_min_cond)) %>%
  full_join(select_common(data_rows_det_min_fre)) %>% 
  full_join(select_common(data_rows_det_min_cond)) %>% 
  full_join(select_common(data_random_min_fre)) %>%
  full_join(select_common(data_random_min_cond)) %>%
  mutate(rating = fre/fre[label == "static_32"])

write_csv(data_mins, "output_merged.csv")

matrix_barplot <- function(data, i, col, ...) {
  title <- sprintf("Matrix %d", i)
  # The order of the plots equals the order in which the tables were merged
  bp_data = data %>% filter(ID == i, label != "static_32") %>% pull(rating)
  #bp_data[bp_data > 2] = NA   # Remove outliers
  barplot(height=bp_data, col=col, log="y", main=title, ...)
}

# Create bar plot for different methods
par(mfrow=c(4,5))
for (i in seq(1,20)) {
  matrix_barplot(data_mins, i, c(10:19))
}
