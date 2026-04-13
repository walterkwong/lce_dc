## Sheet for simulating data using DAG
# From Helen

# Source the functions file
source("LCP_rule_functions.R")

# Load libraries
library(simcausal)
library(reticulate)

# Define variables for simulating data
num_var <- 9 # number of variables in model
n_people <- 1000 # number of people in model
n_time_out <- 40 # number of time points in model
n_time <- n_time_out - 1
set.seed(9)

# Creates DAG
dag <- DAG.empty() +
  node("sex", distr = "rcat.b1", prob = c(0.5, 0.5), replaceNAw0 = TRUE) + ## fem/mal
  node("ethnicity", distr = "rcat.b1", prob = c(0.9, 0.1), replaceNAw0 = TRUE) + ## white/other of mother
  node("marriage", distr = "rcat.b1", prob = c(0.75, 0.25), replaceNAw0 = TRUE) + ## married/single of mother
  node("education", distr = "rcat.b1", prob = c(0.35, 0.65), replaceNAw0 = TRUE) + ## degree or a level/gcse of mother
  node("class", distr = "rcat.b1", prob = c(0.81, 0.19), replaceNAw0 = TRUE) + ## nonmanual/manual of mother
  node("admissions", t = 0:n_time, distr = "rbern", prob = (1 / (5 * log(t + 3)) - 0.015), replaceNAw0 = TRUE) + ## general admissions
  node("admissions2", t = 0:n_time, distr = "rbern", prob = (0.04 * ((class - 1) * 0.28 + 1) * ((education - 1) * 0.22 + 1)), replaceNAw0 = TRUE) + ## something different affected by variables
  node("admissions3", t = 0, distr = "rbern", prob = ((0.02 + sex / 100) * ((ethnicity - 1) * 0.17 + 1) + plogis(-5)), replaceNAw0 = TRUE) + ## general admissions
  node("admissions3", t = 1:n_time, distr = "rbern", prob = ((0.02 + sex / 100) * ((ethnicity - 1) * 0.17 + 1) + plogis(-5 + admissions3[t - 1])), replaceNAw0 = TRUE) + ## general admissions
  node("admissions4", t = 0, distr = "rbern", prob = plogis(-2 + admissions2[t] + admissions[t]), replaceNAw0 = TRUE) + ## something different affected by variables
  node("admissions4", t = 1, distr = "rbern", prob = plogis(-2 + ((admissions2[t] + admissions2[t - 1]) > 0) + ((admissions[t] + admissions[t - 1]) > 0)), replaceNAw0 = TRUE) + ## something different affected by variables
  node("admissions4", t = 2, distr = "rbern", prob = plogis(-2 + ((admissions2[t] + admissions2[t - 1] + admissions2[t - 2]) > 1) + ((admissions[t] + admissions[t - 1] + admissions[t - 2]) > 1)), replaceNAw0 = TRUE) +
  node("admissions4", t = 3:n_time, distr = "rbern", prob = plogis(-2 + ((admissions2[t] + admissions2[t - 1] + admissions2[t - 2] + admissions2[t - 3]) > 2) + ((admissions[t] + admissions[t - 1] + admissions[t - 2] + admissions[t - 3]) > 2)), replaceNAw0 = TRUE) +
  node("Y", t = 0:(n_time - 1), distr = "rbern", prob = plogis(-2), EFU = FALSE, replaceNAw0 = TRUE) +
  node("Y", t = n_time, distr = "rbern", prob = plogis(-1), EFU = FALSE, replaceNAw0 = TRUE)
dag <- set.DAG(dag)

# plot DAG
#plotDAG(dag)
dat_long <- sim(dag, n = n_people)

data <- dat_long[-1]
data_out <- array(dim = c(dim(data)[1], num_var, n_time + 1))
y_out <- array(dim = c(dim(data)[1], n_time + 1))

data_out[, 1, ] <- array(replicate(40, unlist(data[1])), dim = c(dim(data)[1], n_time + 1)) - 1
data_out[, 2, ] <- array(replicate(40, unlist(data[2])), dim = c(dim(data)[1], n_time + 1)) - 1
data_out[, 3, ] <- array(replicate(40, unlist(data[3])), dim = c(dim(data)[1], n_time + 1)) - 1
data_out[, 4, ] <- array(replicate(40, unlist(data[4])), dim = c(dim(data)[1], n_time + 1)) - 1
data_out[, 5, ] <- array(replicate(40, unlist(data[5])), dim = c(dim(data)[1], n_time + 1)) - 1
data_out[, 6, ] <- array(unlist(data[seq(from = 6, by = 5, to = dim(data)[2])]), dim = c(dim(data)[1], n_time + 1))
data_out[, 7, ] <- array(unlist(data[seq(from = 7, by = 5, to = dim(data)[2])]), dim = c(dim(data)[1], n_time + 1))
data_out[, 8, ] <- array(unlist(data[seq(from = 8, by = 5, to = dim(data)[2])]), dim = c(dim(data)[1], n_time + 1))
data_out[, 9, ] <- array(unlist(data[seq(from = 9, by = 5, to = dim(data)[2])]), dim = c(dim(data)[1], n_time + 1))

y_out <- array(unlist(data[seq(from = 10, by = 5, to = dim(data)[2])]), dim = c(dim(data)[1], n_time + 1))
x_out <- data_out

function_names <- c(
  "repeat3", "repeat4", "repeat2_2",
  "order2", "order3", "order4",
  "timing2_0", "timing3_2", "timing4_2", "timing4_4",
  "critical30", "sensitive4", "sensitive5", "weighted4"
)

paper_names <- c(
  "Repeats1", "Repeats2", "Repeats3",
  "Order1", "Order2", "Order3",
  "Timing1", "Timing2", "Timing3", "Timing4",
  "Period1", "Period2", "Period3", "Period4"
)

# moving data to python
# Create data dir
data_dir ="../data/"
if (!file.exists(data_dir)){
  dir.create(file.path(data_dir))}

# create a new environment and install numpy.  Only need to run these lines the first time
#virtualenv_create("r-reticulate")
#virtualenv_install("r-reticulate", "numpy") 

# import Numpy (it will be automatically discovered in "r-reticulate")
use_virtualenv("r-reticulate")
np <- import("numpy") # If you get errors about not being able to find numpy try and restart R

# Saves data
for (i in seq_along(function_names)) {
  print(function_names[i])
  set.seed(9)
  y_outcheck <- eval(parse(text = (paste(function_names[i], "(x_out,y_out)", sep = ""))))

  name <- paste("data_", paper_names[i], sep = "")

  np$save(paste0(data_dir, "/data_X.npy"), r_to_py(x_out))
  np$save(paste0(data_dir, name, "_Y.npy"), r_to_py(y_outcheck))
}