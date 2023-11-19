python inference.py --image_dir /Users/nielsrogge/Documents/python_projecten/table-transformer/images \
--words_dir /Users/nielsrogge/Documents/python_projecten/table-transformer/words \
--mode recognize \
--structure_config_path structure_config.json \
--structure_model_path /Users/nielsrogge/Documents/python_projecten/table-transformer/checkpoints/TATR-v1.1-All-msft.pth \
--structure_device cpu \
--objects \
--visualize \
--csv \
--out_dir output_structure_recognition