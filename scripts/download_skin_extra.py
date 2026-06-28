import kagglehub

path = kagglehub.dataset_download(
    "riyaelizashaju/skin-disease-classification-image-dataset"
)
print("скачано:", path)