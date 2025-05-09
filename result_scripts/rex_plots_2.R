library(data.table)
library(ReX)
library(ggplot2)

path_pcc <- "C:\\Users\\100063082\\Desktop\\VarCoNet_results\\results_HCP\\ReX_files\\PCC_ICC_80_samples.csv"
data_pcc <- fread(path_pcc)
path_varconet <- "C:\\Users\\100063082\\Desktop\\VarCoNet_results\\results_HCP\\ReX_files\\VarCoNet_ICC_80_samples.csv"
data_varconet <- fread(path_varconet)
path_vae <- "C:\\Users\\100063082\\Desktop\\VarCoNet_results\\results_HCP\\ReX_files\\VAE_ICC_80_samples.csv"
data_vae <- fread(path_vae)
path_ae <- "C:\\Users\\100063082\\Desktop\\VarCoNet_results\\results_HCP\\ReX_files\\AE_ICC_80_samples.csv"
data_ae <- fread(path_ae)

data_pcc$group <- "PCC"
data_varconet$group <- "VarCoNet"
data_vae$group <- "(Lu et al., 2024)"
data_ae$group <- "(Cai et al., 2021)"

combined_data <- rbind(data_pcc, data_varconet)
p_varconet_pcc_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3)
ggsave("C:\\Users\\100063082\\Desktop\\VarCoNet-V2\\paper_plots\\VarCoNet_PCC.png")
df_VarPairedComp <- icc_gradient_flow(as.matrix(data_varconet[,"sigma2_w"]), as.matrix(data_varconet[,"sigma2_b"]), as.matrix(data_pcc[,"sigma2_w"]), as.matrix(data_pcc[,"sigma2_b"]))
df_VarPairedComp$contrast <- "VarCoNet-V2 - PCC"
p_varconet_pcc_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
ggsave("C:\\Users\\100063082\\Desktop\\VarCoNet-V2\\paper_plots\\VarCoNet_PCC_grad_norm.png")

combined_data <- rbind(data_vae, data_varconet)
p_varconet_vae_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3)
ggsave("C:\\Users\\100063082\\Desktop\\VarCoNet-V2\\paper_plots\\VarCoNet_VAE.png")
df_VarPairedComp <- icc_gradient_flow(as.matrix(data_varconet[,"sigma2_w"]), as.matrix(data_varconet[,"sigma2_b"]), as.matrix(data_vae[,"sigma2_w"]), as.matrix(data_vae[,"sigma2_b"]))
df_VarPairedComp$contrast <- "VarCoNet-V2 - VAE"
p_varconet_vae_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
ggsave("C:\\Users\\100063082\\Desktop\\VarCoNet-V2\\paper_plots\\VarCoNet_VAE_grad_norm.png")

combined_data <- rbind(data_ae, data_varconet)
p_varconet_ae_1 <- rex_plot.var.field.n(combined_data, size.point = 2, alpha.density = 0.3)
ggsave("C:\\Users\\100063082\\Desktop\\VarCoNet-V2\\paper_plots\\VarCoNet_AE.png")
df_VarPairedComp <- icc_gradient_flow(as.matrix(data_varconet[,"sigma2_w"]), as.matrix(data_varconet[,"sigma2_b"]), as.matrix(data_ae[,"sigma2_w"]), as.matrix(data_ae[,"sigma2_b"]))
df_VarPairedComp$contrast <- "VarCoNet-V2 - AE"
p_varconet_ae_2 <- rex_plot.icc.gradient.norm(df_VarPairedComp,show.contour = FALSE)
ggsave("C:\\Users\\100063082\\Desktop\\VarCoNet-V2\\paper_plots\\VarCoNet_AE_grad_norm.png")


