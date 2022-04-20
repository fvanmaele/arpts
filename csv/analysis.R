library(tidyverse)

select_min_fre <- function(df, ...) {
  df_min <- df %>% 
    group_by(ID) %>% 
    filter(...) %>% 
    slice_min(order_by = fre, with_ties = FALSE) # Only keep the first minimum (multiple may be available)
  return(df_min)
}

# n_halo = 0,1,2
matrix_data <- list(NULL, NULL, NULL)
matrix_data_all <- list()

for (k in seq(0, 2)) {
  for (i in seq(1, 20)) {
    file_name <- sprintf("n_halo_%d/%02d_N512_M16-64_u5_d5_halo%d.csv", k,  i, k)
    matrix_data[[k+1]][[i]] = read_csv(file_name)
  }
  matrix_data_all[[k+1]] <- bind_rows(matrix_data[[k+1]])
}

# Process table of all matrix IDs
min_fre_static_partition <- select_min_fre(matrix_data_all[[1]], k_max_up == 0, k_max_down == 0)

# Minimum FRE for partition with shifted boundaries (includes k_max_up=0, k_max_down=0)
min_fre_shifted_partition_halo_0 <- select_min_fre(matrix_data_all[[1]])
min_fre_shifted_partition_halo_1 <- select_min_fre(matrix_data_all[[2]])
min_fre_shifted_partition_halo_2 <- select_min_fre(matrix_data_all[[3]])

# Create bar plot of minimum FRE for different values of n_halo
barplot_data <- matrix(nrow=4, ncol=20)
barplot_data[1,] <- min_fre_static_partition[["fre"]]
barplot_data[2,] <- min_fre_shifted_partition_halo_0[["fre"]]
barplot_data[3,] <- min_fre_shifted_partition_halo_1[["fre"]]
barplot_data[4,] <- min_fre_shifted_partition_halo_2[["fre"]]

par(mfrow=c(4,5))
for (i in seq(1,20)) {
  title <- sprintf("Matrix %d", i)
  barplot(height=barplot_data[,i], col=c(1:4), log="y", main=title, legend.text=T)
}
#legend("topright", legend=c("S", "H0", "H1", "H2"), fill=c(1:4))
