# load the libraries
library(sits)
library(tibble)
library(sf)
library(xml2)
library(dplyr)
library(torch)
library(torchopt)
library(sp)
library(raster)
library(httr)

#access key for BDC
Sys.setenv("BDC_ACCESS_KEY" = "xxxxx")

# Leia os shapefiles de pontos e áreas
recorte <- st_read("./Qgis/recorte/recorteTresPontas.shp")
teste <- st_read("./Qgis/teste.geojson")
train <- st_read("./Qgis/train.geojson")

teste_noLabel <- subset(teste, select = -c(Shape_Area, label, Shape_Leng, id))

#create date cube for Três Pontas tile
sentinel_TP <- sits_cube(
  source = "BDC",
  collection = "SENTINEL-2-16D",
  roi = recorte,
  bands= c("EVI", "NDVI", "B02", "B03", "B04", "B08", "CLOUD"),
  start_date = "2017-01-01",
  end_date = "2018-12-31"
)

#Check cube dates
sits_timeline(sentinel_TP)

# relates cube to the samples
series_train <- sits_get_data(
  cube = sentinel_TP,
  samples = train,
)

series_teste <- sits_get_data(
  cube = sentinel_TP,
  samples = teste,
)

#--------------Training algorithm-----------------

rfor_model <- sits_train(
  samples = series_train,
  ml_method = sits_rfor()
)

tempcnn_model <- sits_train(
  series_train,
  sits_tempcnn(
    optimizer            = torchopt::optim_adamw,
    cnn_layers           = c(128, 128, 128),
    cnn_kernels          = c(7, 7, 7),
    cnn_dropout_rates    = c(0.2, 0.2, 0.2),
    epochs               = 100,
    batch_size           = 64,
    validation_split     = 0.2,
    verbose              = FALSE
  )
)

#----------Classify--------------
#create test cube
teste_cube <- sits_cube(
  source = "BDC",
  collection = "SENTINEL-2-16D",
  bands= c("EVI", "NDVI", "B02", "B03", "B04", "B08", "CLOUD"),
  roi = recorte,
  start_date = "2017-01-01",
  end_date = "2018-12-31"
)

tempo_execucao_rf <- system.time({
  cube_TP_probs_rf <- sits_classify(
    data = teste_cube,
    ml_model = rfor_model,
    roi = recorte,
    output_dir = "./caminhoDoArquivoParaSalvar",
    version = "rf-probs-roi",
    multicores = 10,
    memsize = 16
  )
})

cube_TP_rf <- sits_classify(
  data = series_teste,
  ml_model = rfor_model,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "rf-probs-series",
  multicores = 10,
  memsize = 16
)

tempo_execucao_tempcnn <- system.time({
  cube_TP_probs_tempcnn <- sits_classify(
    data = sentinel_TP,
    ml_model = tempcnn_model,
    roi = recorte,
    output_dir = "./caminhoDoArquivoParaSalvar",
    version = "tempcnn-probs-roi",
    multicores = 10,
    memsize = 16
  )
})

cube_TP_tempcnn <- sits_classify(
  data = series_teste,
  ml_model = tempcnn_model,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-probs-series",
  multicores = 10,
  memsize = 16
)

#--------Perform a five-fold validation for the dataset-----------
val_rfor <- sits_kfold_validate(
  samples = series_train,
  folds = 5,
  ml_method = sits_rfor(),
  multicores = 5
)

val_tcnn <- sits_kfold_validate(
  samples = series_train,
  ml_method = sits_tempcnn(
    optimizer = torchopt::optim_adamw,
    opt_hparams = list(lr = 0.001)
  ),
  folds = 5,
  multicores = 5
)

# Print the validation statistics
summary(val_rfor)

summary(val_tcnn)

#-----------------Generate a thematic map-----------------------
cafe_class <- sits_label_classification(
  cube = cube_TP_probs_rf,
  multicores = 4,
  memsize = 12,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "rf-cafe-class"
)

cafe_class_series <- sits_label_classification(
  cube = cube_TP_rf,
  multicores = 4,
  memsize = 12,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "rf-cafe-class-series"
)

cafe_class_tempcnn <- sits_label_classification(
  cube = cube_TP_probs_tempcnn,
  multicores = 4,
  memsize = 12,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-cafe-class"
)

labels<-c("1" = "café", "2" = "não_café")

# Plot the thematic map
plot(cafe_class,
     tmap_options = list("legend_text_size" = 0.7))

#----------------Accuracy-----------------------------------
rf_acc <- sits_accuracy(cube_TP_rf, validation = series_teste)
rf_acc

tempcnn_acc <- sits_accuracy(cube_TP_tempcnn, validation = series_teste)
tempcnn_acc

#------------------Other index---------------------------
sentinel_TP <- sits_apply(sentinel_TP,
                          NDWI = (B03 - B08) / (B03 + B08),
                          output_dir = "./caminhoDoArquivoParaSalvar/index/NDWI")

sentinel_TP <- sits_apply(sentinel_TP,
                          VARI = (B03 - B04) / (B03 + B04 - B02),
                          output_dir = "./caminhoDoArquivoParaSalvar/index/VARI")

sentinel_TP <- sits_apply(sentinel_TP,
                          WI = (B08) / (B03),
                          output_dir = "./caminhoDoArquivoParaSalvar/index/WI")

sentinel_TP <- sits_apply(sentinel_TP,
                          CVI = (B08 / B04) * (B03 / B04),
                          output_dir = "./caminhoDoArquivoParaSalvar/index/CVI")

sentinel_TP <- sits_apply(sentinel_TP,
                          MSAVI = (2 * B08 + 1 - sqrt((2 * B08 +1 )^2 - 8 * (B08 - B04))) / 2,
                          output_dir = "./caminhoDoArquivoParaSalvar/index/MSAVI")

sits_bands(reg_cube)

#--------------Training algorithm with new index-----------------

# relates cube to the samples
series_train_index <- sits_get_data(
  cube = sentinel_TP,
  samples = train
)

series_teste_index <- sits_get_data(
  cube = sentinel_TP,
  samples = teste
)

rfor_model_index <- sits_train(
  samples = series_train_index,
  ml_method = sits_rfor()
)

tempcnn_model_index <- sits_train(
  series_train_index,
  sits_tempcnn(
    optimizer            = torchopt::optim_adamw,
    cnn_layers           = c(128, 128, 128),
    cnn_kernels          = c(7, 7, 7),
    cnn_dropout_rates    = c(0.2, 0.2, 0.2),
    epochs               = 100,
    batch_size           = 64,
    validation_split     = 0.2,
    verbose              = FALSE
  )
)

#----------Classify with new index--------------

tempo_execucao_rf_index <- system.time({
  cube_TP_probs_rf_index <- sits_classify(
    data = sentinel_TP,
    ml_model = rfor_model_index,
    roi = recorte,
    output_dir = "./caminhoDoArquivoParaSalvar",
    version = "rf-index-roi",
    multicores = 10,
    memsize = 16
  )
})

cube_TP_rf_index <- sits_classify(
  data = series_teste_index,
  ml_model = rfor_model_index,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "rf-series-index",
  multicores = 10,
  memsize = 16
)

tempo_execucao_tempcnn_index <- system.time({
  cube_TP_probs_tempcnn_index <- sits_classify(
    data = sentinel_TP,
    ml_model = tempcnn_model_index,
    roi = recorte,
    output_dir = "./caminhoDoArquivoParaSalvar",
    version = "tempcnn-index-roi",
    multicores = 4,
    memsize = 16
  )
})

cube_TP_tempcnn_index <- sits_classify(
  data = series_teste_index,
  ml_model = tempcnn_model_index,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-series-index",
  multicores = 10,
  memsize = 16
)

#----------------Accuracy-----------------------------------

cube_TP_rf_index$label <- gsub("não café", "não.café", cube_TP_rf_index$label)
cube_TP_tempcnn_index$label <- gsub("não café", "não.café", cube_TP_tempcnn_index$label)

rf_acc_index <- sits_accuracy(cube_TP_rf_index, validation = series_teste_index)
rf_acc_index

tempcnn_acc_index <- sits_accuracy(cube_TP_tempcnn_index, validation = series_teste_index)
tempcnn_acc_index

#-----------------Generate a thematic map with index-----------------------
cafe_class_index <- sits_label_classification(
  cube = cube_TP_probs_rf_index,
  multicores = 4,
  memsize = 12,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "rf-cafe-class-index"
)

cafe_class_tempcnn_index <- sits_label_classification(
  cube = cube_TP_probs_tempcnn_index,
  multicores = 4,
  memsize = 12,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-cafe-class-index"
)

#----------------Map TP 2022-----------------------------------
sentinel_TP_2024 <- sits_cube(
  source = "BDC",
  collection = "SENTINEL-2-16D",
  roi = recorte,
  bands= c("EVI", "NDVI", "B02", "B03", "B04", "B08", "CLOUD"),
  start_date = "2020-10-15",
  end_date = "2022-09-30"
)
summary(sentinel_TP_2024)

sentinel_TP_2024 <- sits_apply(sentinel_TP_2024,
                          MSAVI = (2 * B08 + 1 - sqrt((2 * B08 +1 )^2 - 8 * (B08 - B04))) / 2,
                          output_dir = "./caminhoDoArquivoParaSalvar/index_2022/MSAVI")

sentinel_TP_2024 <- sits_apply(sentinel_TP_2024,
                          NDWI = (B03 - B08) / (B03 + B08),
                          output_dir = "./caminhoDoArquivoParaSalvar/index_2022/NDWI")

sentinel_TP_2024 <- sits_apply(sentinel_TP_2024,
                          VARI = (B03 - B04) / (B03 + B04 - B02),
                          output_dir = "./caminhoDoArquivoParaSalvar/index_2022/VARI")

cube_TP_probs_tempcnn_index_2022 <- sits_classify(
  data = sentinel_TP_2024,
  ml_model = tempcnn_model_index,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-index-2022",
  multicores = 14,
  memsize = 16
)


cafe_class_tempcnn_index_2022 <- sits_label_classification(
  cube = cube_TP_probs_tempcnn_index_2022,
  multicores = 4,
  memsize = 12,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-cafe-class-index-2022"
)

#----------------Plots-------------------------

#Patterns associated to the training samples
series_teste |>
  sits_select(bands = c("NDVI", "EVI", "B08")) |>
  sits_patterns() |>
  plot()

plot(rfor_model)
plot(rfor_model_index)


plot(tempcnn_model)
plot(tempcnn_model_index)

plot(rfor_model)
