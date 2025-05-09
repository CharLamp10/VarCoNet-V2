library(data.table)
library(ReX)
library(ggplot2)

#PCC
path_pcc <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\rex_AICHA_PCC_80_samples.csv"
data_pcc <- fread(path_pcc)
x <- data_pcc[, 4:ncol(data_pcc)]
sub <- data_pcc[,2]
sess <- data_pcc[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_pcc <- data.frame(lme_ICC_1wayR(x, sub, sess))
p_pcc <- rex_plot.var.field(df_icc_pcc, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
ggsave("C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\PCC_ICC_80_samples.png", plot = p_pcc)
write.csv(df_icc_pcc, file = "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\PCC_ICC_80_samples.csv", row.names = FALSE)

#VarCoNetV2
path_varconet <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\rex_AICHA_VarCoNetV2_80_samples.csv"
data_varconet <- fread(path_varconet)
x <- data_varconet[, 4:ncol(data_varconet)]
sub <- data_varconet[,2]
sess <- data_varconet[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_varconet <- data.frame(lme_ICC_1wayR(x, sub, sess))
p_varconet <- rex_plot.var.field(df_icc_varconet, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
ggsave("C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VarCoNetV2_ICC_80_samples.png", plot = p_varconet)
write.csv(df_icc_varconet, file = "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VarCoNetV2_ICC_80_samples.csv", row.names = FALSE)

#VAE
path_vae <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\rex_AICHA_VAE_80_samples.csv"
data_vae <- fread(path_vae)
x <- data_vae[, 4:ncol(data_vae)]
sub <- data_vae[,2]
sess <- data_vae[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_vae <- data.frame(lme_ICC_1wayR(x, sub, sess))
p_vae <- rex_plot.var.field(df_icc_vae, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
ggsave("C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VAE_ICC_80_samples.png", plot = p_vae)
write.csv(df_icc_vae, file = "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VAE_ICC_80_samples.csv", row.names = FALSE)


#AE
path_ae <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\rex_AICHA_AE_80_samples.csv"
data_ae <- fread(path_ae)
x <- data_ae[, 4:ncol(data_ae)]
sub <- data_ae[,2]
sess <- data_ae[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_ae <- data.frame(lme_ICC_1wayR(x, sub, sess))
p_ae <- rex_plot.var.field(df_icc_ae, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
ggsave("C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\AE_ICC_80_samples.png", plot = p_ae)
write.csv(df_icc_ae, file = "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\AE_ICC_80_samples.csv", row.names = FALSE)


#PFN
path_pfn <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\rex_AICHA_PFN_80_samples.csv"
data_pfn <- fread(path_pfn)
x <- data_pfn[, 4:ncol(data_pfn)]
sub <- data_pfn[,2]
sess <- data_pfn[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_pfn <- data.frame(lme_ICC_1wayR(x, sub, sess))
p_pfn <- rex_plot.var.field(df_icc_pfn, size.point = 2, alpha.density = 0.3, color.point.fill = "red", color = "red")
ggsave("C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\PFN_ICC_80_samples.png", plot = p_pfn)
write.csv(df_icc_pfn, file = "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\PFN_ICC_80_samples.csv", row.names = FALSE)

#VarCoNetV2
path_varconet <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\rex_AICHA_VarCoNetV2_200_samples.csv"
data_varconet <- fread(path_varconet)
x <- data_varconet[, 4:ncol(data_varconet)]
sub <- data_varconet[,2]
sess <- data_varconet[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_varconet <- data.frame(lme_ICC_1wayR(x, sub, sess))
write.csv(df_icc_varconet, file = "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VarCoNetV2_AICHA_ICC_200_samples.csv", row.names = FALSE)

#VarCoNetV2
path_varconet <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\rex_AICHA_VarCoNetV2_320_samples.csv"
data_varconet <- fread(path_varconet)
x <- data_varconet[, 4:ncol(data_varconet)]
sub <- data_varconet[,2]
sess <- data_varconet[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_varconet <- data.frame(lme_ICC_1wayR(x, sub, sess))
write.csv(df_icc_varconet, file = "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VarCoNetV2_AICHA_ICC_320_samples.csv", row.names = FALSE)

#VarCoNetV2
path_varconet <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\rex_AAL_VarCoNetV2_80_samples.csv"
data_varconet <- fread(path_varconet)
x <- data_varconet[, 4:ncol(data_varconet)]
sub <- data_varconet[,2]
sess <- data_varconet[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_varconet <- data.frame(lme_ICC_1wayR(x, sub, sess))
write.csv(df_icc_varconet, file = "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VarCoNetV2_AAL_ICC_80_samples.csv", row.names = FALSE)

#VarCoNetV2
path_varconet <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\rex_AAL_VarCoNetV2_200_samples.csv"
data_varconet <- fread(path_varconet)
x <- data_varconet[, 4:ncol(data_varconet)]
sub <- data_varconet[,2]
sess <- data_varconet[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_varconet <- data.frame(lme_ICC_1wayR(x, sub, sess))
write.csv(df_icc_varconet, file = "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VarCoNetV2_AAL_ICC_200_samples.csv", row.names = FALSE)

#VarCoNetV2
path_varconet <- "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\rex_AAL_VarCoNetV2_320_samples.csv"
data_varconet <- fread(path_varconet)
x <- data_varconet[, 4:ncol(data_varconet)]
sub <- data_varconet[,2]
sess <- data_varconet[,3]
x <- as.matrix(x)
sub <- as.matrix(sub)
sess <- as.matrix(sess)
df_icc_varconet <- data.frame(lme_ICC_1wayR(x, sub, sess))
write.csv(df_icc_varconet, file = "C:\\Users\\100063082\\Desktop\\VarCoNetV2\\rex_results\\VarCoNetV2_AAL_ICC_320_samples.csv", row.names = FALSE)



