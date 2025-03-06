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

#access key for BDC
Sys.setenv("BDC_ACCESS_KEY" = "xxxxx")

# Leia os shapefiles de pontos e áreas de Carmo de Minas
recorte <- st_read("./Qgis/recorte/recorte.shp")
teste <- st_read("./Qgis/teste.geojson")
train <- st_read("./Qgis/train.geojson")

teste_noLabel <- subset(teste, select = -c(AREA_ha, label, OBJECTID))

#create date cube for carmo de minas tile
sentinel_CM <- sits_cube(
  source = "BDC",
  collection = "SENTINEL-2-16D",
  tiles = "031029",
  bands= c("EVI", "NDVI", "B02", "B03", "B04", "B08", "CLOUD"),
  start_date = "2017-01-01",
  end_date = "2018-12-31"
)

#Check cube dates
sits_timeline(sentinel_CM)

# relates cube to the samples
series_train <- sits_get_data(
  cube = sentinel_CM,
  samples = train,
)

series_teste <- sits_get_data(
  cube = sentinel_CM,
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
  roi = teste_noLabel,
  start_date = "2017-01-01",
  end_date = "2018-12-31"
)

cube_CM_probs_rf <- sits_classify(
  data = teste_cube,
  ml_model = rfor_model,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "rf-2205",
  multicores = 10,
  memsize = 16
)

cube_CM_rf <- sits_classify(
  data = series_teste,
  ml_model = rfor_model,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "rf-2205-series",
  multicores = 10,
  memsize = 16
)

cube_CM_probs_tempcnn <- sits_classify(
  data = teste_cube,
  ml_model = tempcnn_model,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-2205",
  multicores = 10,
  memsize = 16
)

cube_CM_tempcnn <- sits_classify(
  data = series_teste,
  ml_model = tempcnn_model,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-2205-series",
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
  cube = cube_CM_probs_rf,
  multicores = 4,
  memsize = 12,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "rf-cafe-class"
)

cafe_class_series <- sits_label_classification(
  cube = cube_CM_probs_rf,
  multicores = 4,
  memsize = 12,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "rf-cafe-class-series"
)

cafe_class_tempcnn <- sits_label_classification(
  cube = cube_CM_probs_tempcnn,
  multicores = 4,
  memsize = 12,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-cafe-class"
)

labels<-c("1" = "café", "2" = "não_café")

# Plot the thematic map
plot(cafe_class,
     tmap_options = list("legend_text_size" = 0.7))

#--------------- put labels in classify cube----------------

cafe_class <- sits_cube(
  source = "BDC",
  collection = "SENTINEL-2-16D",
  bands= "class",
  labels = labels,
  data_dir = "/home/pietrasantos/Desktop/fixCode/",
  parse_info = c("X1", "tile", "band", "start_date", "end_date", "version"),
  progress = FALSE
)


#----------------Accuracy-----------------------------------

cube_CM_rf$label <- gsub("não café", "não.café", cube_CM_rf$label)
cube_CM_tempcnn$label <- gsub("não café", "não.café", cube_CM_tempcnn$label)

rf_acc <- sits_accuracy(cube_CM_rf, validation = series_teste)
rf_acc

tempcnn_acc <- sits_accuracy(cube_CM_tempcnn, validation = series_teste)
tempcnn_acc

#------------------Other index---------------------------
sentinel_CM <- sits_apply(sentinel_CM,
                       NDWI = (B03 - B08) / (B03 + B08),
                       output_dir = "./caminhoDoArquivoParaSalvar/index/NDWI")

sentinel_CM <- sits_apply(sentinel_CM,
                       VARI = (B03 - B04) / (B03 + B04 - B02),
                       output_dir = "./caminhoDoArquivoParaSalvar/index/VARI")

sentinel_CM <- sits_apply(sentinel_CM,
                       WI = (B08) / (B03),
                       output_dir = "./caminhoDoArquivoParaSalvar/index/WI")

sentinel_CM <- sits_apply(sentinel_CM,
                       CVI = (B08 / B04) * (B03 / B04),
                       output_dir = "./caminhoDoArquivoParaSalvar/index/CVI")

sentinel_CM <- sits_apply(sentinel_CM,
                       MSAVI = (2 * B08 + 1 - sqrt((2 * B08 +1 )^2 - 8 * (B08 - B04))) / 2,
                       output_dir = "./caminhoDoArquivoParaSalvar/index/MSAVI")

sits_bands(reg_cube)

#--------------Training algorithm with new index-----------------

# relates cube to the samples
series_train_index <- sits_get_data(
  cube = sentinel_CM,
  samples = train
)

series_teste_index <- sits_get_data(
  cube = sentinel_CM,
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

cube_CM_probs_rf_index <- sits_classify(
  data = sentinel_CM,
  ml_model = rfor_model_index,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "rf-index",
  multicores = 10,
  memsize = 16
)

cube_CM_rf_series_index <- sits_classify(
  data = series_teste_index,
  ml_model = rfor_model_index,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "rf-series-index",
  multicores = 10,
  memsize = 16
)

cube_CM_probs_tempcnn_index <- sits_classify(
  data = sentinel_CM,
  ml_model = tempcnn_model_index,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-index-2705",
  multicores = 10,
  memsize = 16
)

cube_CM_tempcnn_series_index <- sits_classify(
  data = series_teste_index,
  ml_model = tempcnn_model_index,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-series-index",
  multicores = 10,
  memsize = 16
)

#----------------Accuracy-----------------------------------

cube_CM_rf_series_index$label <- gsub("não café", "não.café", cube_CM_rf_series_index$label)
cube_CM_tempcnn_series_index$label <- gsub("não café", "não.café", cube_CM_tempcnn_series_index$label)

rf_acc_index <- sits_accuracy(cube_CM_rf_series_index, validation = series_teste_index)
rf_acc_index

tempcnn_acc_index <- sits_accuracy(cube_CM_tempcnn_series_index, validation = series_teste_index)
tempcnn_acc_index

#-----------------Generate a thematic map with index-----------------------
cafe_class_index <- sits_label_classification(
  cube = cube_CM_probs_rf_index,
  multicores = 4,
  memsize = 12,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "rf-cafe-class-index"
)

cafe_class_series <- sits_label_classification(
  cube = cube_CM_rf_index,
  multicores = 4,
  memsize = 12,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "rf-cafe-class-series-index"
)

cafe_class_tempcnn_index <- sits_label_classification(
  cube = cube_CM_probs_tempcnn_index,
  multicores = 4,
  memsize = 12,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-cafe-class-index"
)

#-----------Map CM 2022------------------------
#create date cube for carmo de minas tile
sentinel_CM_2022 <- sits_cube(
  source = "BDC",
  collection = "SENTINEL-2-16D",
  tiles = "031029",
  bands= c("EVI", "NDVI", "B02", "B03", "B04", "B08", "CLOUD"),
  start_date = "2020-10-15",
  end_date = "2022-09-30"
)

tempcnn_model_CM <- sits_train(
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

cube_CM_probs_tempcnn_2022 <- sits_classify(
  data = sentinel_CM_2022,
  ml_model = tempcnn_model_CM,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-2022-2",
  multicores = 4,
  memsize = 12
)

cafe_class_2022 <- sits_label_classification(
  cube = cube_CM_probs_tempcnn_2022,
  multicores = 4,
  memsize = 12,
  output_dir = "./caminhoDoArquivoParaSalvar",
  version = "tempcnn-cafe-class-2022"
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

# Obtain a time series from the raster cube from a point
sample_latlong <- tibble::tibble(
  longitude = -45.14651995938565,
  latitude  = -22.095694485906652
)

point <- sits_get_data(
  cube = sentinel_CM,
  samples = sample_latlong
)

