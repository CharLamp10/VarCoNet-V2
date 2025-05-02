library(rsfcNet)
library(RcppCNPy)
library(data.table)


path_varconet_aicha_80 <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VarCoNetV2_ICC_80_samples.csv"
data_varconet_aicha_80 <- as.matrix(fread(path_varconet_aicha_80)[["ICC"]])
path_varconet_aicha_200 <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VarCoNetV2_AICHA_ICC_200_samples.csv"
data_varconet_aicha_200 <- as.matrix(fread(path_varconet_aicha_200)[["ICC"]])
path_varconet_aicha_320 <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VarCoNetV2_AICHA_ICC_320_samples.csv"
data_varconet_aicha_320 <- as.matrix(fread(path_varconet_aicha_320)[["ICC"]])
path_varconet_aal_80 <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VarCoNetV2_AAL_ICC_80_samples.csv"
data_varconet_aal_80 <- as.matrix(fread(path_varconet_aal_80)[["ICC"]])
path_varconet_aal_200 <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VarCoNetV2_AAL_ICC_200_samples.csv"
data_varconet_aal_200 <- as.matrix(fread(path_varconet_aal_200)[["ICC"]])
path_varconet_aal_320 <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VarCoNetV2_AAL_ICC_320_samples.csv"
data_varconet_aal_320 <- as.matrix(fread(path_varconet_aal_320)[["ICC"]])

data_varconet_aicha = (data_varconet_aicha_80 + data_varconet_aicha_200 + data_varconet_aicha_320)/3
data_varconet_aal = (data_varconet_aal_80 + data_varconet_aal_200 + data_varconet_aal_320)/3

top_values <- sort(data_varconet_aicha, decreasing = TRUE)[1:50]
data_varconet_aicha <- ifelse(data_varconet_aicha %in% top_values, data_varconet_aicha, 0)

top_values <- sort(data_varconet_aal, decreasing = TRUE)[1:50]
data_varconet_aal <- ifelse(data_varconet_aal %in% top_values, data_varconet_aal, 0)

mat_varconet_aicha <- matrix(0, nrow = 384, ncol = 384)
tri_idx <- which(upper.tri(mat_varconet_aicha))
mat_varconet_aicha[tri_idx] <- data_varconet_aicha
mat_varconet_aicha <- mat_varconet_aicha + t(mat_varconet_aicha)

mat_varconet_aal <- matrix(0, nrow = 166, ncol = 166)
tri_idx <- which(upper.tri(mat_varconet_aal))
mat_varconet_aal[tri_idx] <- data_varconet_aal
mat_varconet_aal <- mat_varconet_aal + t(mat_varconet_aal)

aal_coords <- npyLoad("C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\AAL3_coords.npy")
aicha_coords <- npyLoad("C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\AICHA_coords.npy")

nonzero_rows <- apply(mat_varconet_aicha, 1, function(row) any(row != 0))
row_indices <- which(nonzero_rows)
aicha_coords <- aicha_coords[nonzero_rows, , drop = FALSE]
mat_varconet_aicha <- mat_varconet_aicha[nonzero_rows,nonzero_rows]

nonzero_rows <- apply(mat_varconet_aal, 1, function(row) any(row != 0))
row_indices <- which(nonzero_rows)
aal_coords <- aal_coords[nonzero_rows, , drop = FALSE]
mat_varconet_aal <- mat_varconet_aal[nonzero_rows,nonzero_rows]

write_brainNet(
  aicha_coords,
  node.size = rep(0, length(aicha_coords)),
  mat_varconet_aicha,
  directory = 'C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\',
  name = 'brainNet_AICHA'
)

write_brainNet(
  aal_coords,
  node.size = rep(0, length(aal_coords)),
  mat_varconet_aal,
  directory = 'C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\',
  name = 'brainNet_AAL'
)


path_varconet_aal_abide <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\feature_importance\\AAL_feature_importance.csv"
data_varconet_aal_abide <- as.matrix(fread(path_varconet_aal_abide))
path_varconet_aicha_abide <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\feature_importance\\AICHA_feature_importance.csv"
data_varconet_aicha_abide <- as.matrix(fread(path_varconet_aicha_abide))

top_values <- sort(data_varconet_aal_abide, decreasing = TRUE)[1:20]
data_varconet_aal_abide <- ifelse(data_varconet_aal_abide %in% top_values, data_varconet_aal_abide, 0)

top_values <- sort(data_varconet_aicha_abide, decreasing = TRUE)[1:20]
data_varconet_aicha_abide <- ifelse(data_varconet_aicha_abide %in% top_values, data_varconet_aicha_abide, 0)

mat_varconet_aicha <- matrix(0, nrow = 384, ncol = 384)
tri_idx <- which(upper.tri(mat_varconet_aicha))
mat_varconet_aicha[tri_idx] <- data_varconet_aicha_abide
mat_varconet_aicha <- mat_varconet_aicha + t(mat_varconet_aicha)

mat_varconet_aal <- matrix(0, nrow = 166, ncol = 166)
tri_idx <- which(upper.tri(mat_varconet_aal))
mat_varconet_aal[tri_idx] <- data_varconet_aal_abide
mat_varconet_aal <- mat_varconet_aal + t(mat_varconet_aal)

aal_coords <- npyLoad("C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\AAL3_coords.npy")
aicha_coords <- npyLoad("C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\AICHA_coords.npy")

nonzero_rows <- apply(mat_varconet_aicha, 1, function(row) any(row != 0))
row_indices <- which(nonzero_rows)
aicha_coords <- aicha_coords[nonzero_rows, , drop = FALSE]
mat_varconet_aicha <- mat_varconet_aicha[nonzero_rows,nonzero_rows]

nonzero_rows <- apply(mat_varconet_aal, 1, function(row) any(row != 0))
row_indices <- which(nonzero_rows)
aal_coords <- aal_coords[nonzero_rows, , drop = FALSE]
mat_varconet_aal <- mat_varconet_aal[nonzero_rows,nonzero_rows]

write_brainNet(
  aicha_coords,
  node.size = rep(0, length(aicha_coords)),
  mat_varconet_aicha,
  directory = 'C:\\Users\\100063082\\Desktop\\VarCoNetV2\\feature_importance\\',
  name = 'brainNet_AICHA_ASD'
)

write_brainNet(
  aal_coords,
  node.size = rep(0, length(aal_coords)),
  mat_varconet_aal,
  directory = 'C:\\Users\\100063082\\Desktop\\VarCoNetV2\\feature_importance\\',
  name = 'brainNet_AAL_ASD'
)


