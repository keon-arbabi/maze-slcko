library(tidyverse)
setwd("projects/def-wainberg/karbabi/maze-slcko")

contrasts = c(
    "WT_AAI_vs_WT", "SLCKO_AAI_vs_SLCKO", "WT_DAPA_vs_WT", 
    "SLCKO_vs_WT", "SLCKO_AAI_vs_WT_AAI", "WT_AAI_DAPA_vs_WT_AAI")

de_table = do.call(rbind, 
    lapply(contrasts, function(c) {
        read_csv(paste0("output/DE_", c, ".csv")) %>%
            mutate(contrast = c)}))

de_table %>%  
    mutate(Dir = factor(sign(logFC)), cell_type = as.factor(cell_type)) %>%
    mutate(Dir = recode_factor(Dir, `-1` = "Down", `1` = "Up")) %>%
    group_by(contrast, cell_type, Dir) %>% 
    dplyr::summarize(
        "pt01" = sum(FDR < 0.01),
        "pt05" = sum(FDR < 0.05),
        "pt10" = sum(FDR < 0.10)) %>%
    ungroup() %>%
    pivot_longer(cols = c(pt01, pt05, pt10),
                values_to = "Freq",
                names_to = "FDR_thresh") %>%
    mutate(Freq = if_else(Dir == "Down", -Freq, Freq)) %>%
    mutate(Freq = if_else(Freq == 0, NA, Freq)) %>%
    filter(!is.na(Freq)) %>%
    mutate(contrast = str_replace_all(contrast, "_", " ")) %>%
ggplot(., aes(x = reorder(cell_type, abs(Freq)), y = Freq, fill = Dir)) +
    geom_bar(data = . %>% filter(FDR_thresh == "pt01"),
        stat = "identity", width = 0.8, alpha = 1) +
    geom_bar(data = . %>% filter(FDR_thresh == "pt05"), 
        stat = "identity", width = 0.8, alpha = 0.7) +
    geom_bar(data = . %>% filter(FDR_thresh == "pt10"),
        stat = "identity", width = 0.8, alpha = 0.5) +
    geom_text(data = . %>% filter(FDR_thresh == "pt10"), 
            aes(label = abs(Freq), y = Freq-5), 
            hjust = -0.1, size = 3) +
    geom_tile(aes(y = NA_integer_, alpha = factor(FDR_thresh))) + 
    scale_fill_manual(values = c("#1F77B4FF", "#D62728FF")) +
    scale_alpha_manual("FDR Threshold", values = c(1, 0.7, 0.5), 
        labels = c("0.01","0.05","0.10")) + 
    labs(y = "Number of DE genes", x = "", fill = "Direction") +
    coord_flip() +
    facet_wrap(~ contrast, ncol = 3, drop=F) +
    theme_classic() +
      theme(strip.text = element_text(size = 12, color = "black"),
        axis.text = element_text(size = 12, color = "black"))

ggsave("figures/number_of_degs.png", height = 8, width = 12)

library(clusterProfiler)
library(org.Mm.eg.db)

plot_list = list()
for(c in unique(de_table$contrast)){
    for(ct in unique(de_table$cell_type)){
        gsea_df = de_table %>% 
            dplyr::filter(contrast == c, cell_type == ct) %>%
            dplyr::mutate(rank = sign(logFC) * -log10(P)) %>%
            dplyr::arrange(desc(rank))
        gsea_v = gsea_df$rank
        names(gsea_v) = gsea_df$gene
        
        print(paste(c, ct))
        gsea = gseGO(geneList = gsea_v, 
            ont = "BP", 
            keyType = "SYMBOL", 
            minGSSize = 10, 
            maxGSSize = 500, 
            pvalueCutoff = 0.1, 
            verbose = TRUE, 
            OrgDb = org.Mm.eg.db, 
            pAdjustMethod = "fdr")
        
        #gsea = simplify(gse, cutoff = 0.7, by = "p.adjust", select_fun = min)
        
        if(dim(gsea@result)[1] > 5) {
            plot_list[[paste(c, ct)]] =
                enrichplot::dotplot(gsea, showCategory = 15, split = ".sign") +
                    facet_grid(~.sign) +
                    scale_y_discrete(labels = function(x) 
                        str_wrap(x, width = 50))
        }
    }
}
png("/figures/tmp.png"), height = 9, width = 8, units = "in", res = 300)
ggpubr::ggarrange(plotlist = plot_list)
dev.off()