input_all = read_csv('output_min_coarse_cond.csv')

input_min <- input_all %>% 
  group_by(ID) %>% 
  slice_min(order_by = fre, with_ties = FALSE)
