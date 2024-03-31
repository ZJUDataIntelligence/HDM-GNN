# HDM-GNN
This is the Pytorch implementation of HDM-GNN in the paper: [HDM-GNN: A Heterogeneous Dynamic Multi-view Graph Neural Network for Crime Prediction] .

## Requirements
- pytorch
- pytorch-geometric
- geopandas

## Usage
To run the code
1. cd to *data* folder:
`unzip data.zip`
2. cd to *model* folder:
`python train.py --config ../config/config.json --output ../output/`

## To handle the raw datasets

1. Mapping data with geographic locations to area id. Please refer to the code in this /data_process/node_map, which may need some minor adjustments depending on your specific data format.
2. Constructing node features based on the mapped POI data, crime data and 311 data. Please refer to the code in this /data_process/node_extraction, which may need some minor adjustments depending on your specific data format.
3. Constructing edges based on the mapped mobility data and function similarity data. Function similarity is calculated based on POI node features. Please refer to the code in this /data_process/node_extraction, which may need some minor adjustments depending on your specific data format.
4. Train and test the model with the codes in /model.

