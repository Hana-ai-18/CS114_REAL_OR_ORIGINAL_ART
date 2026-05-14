import kagglehub

# Download latest version
path = kagglehub.dataset_download("ravidussilva/ai-artbench", path="./dataset")

print("Path to dataset files:", path)