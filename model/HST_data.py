from torch_geometric.data import HeteroData
from torch_geometric.data import Data


class HSTData(Data):
    # hetero_data_list is a list of torch_geometric.data.HeteroData()
    # the length of the list should be TIME_SERIES_LENGTH
    def __init__(self, hetero_data_list):
        super().__init__()
        self.hetero_data_list = hetero_data_list
        self.y = -1

class WholeData(Data):
    # hetero_data_list is a list of torch_geometric.data.HeteroData()
    # the length of the list should be TIME_SERIES_LENGTH
    def __init__(self, hst_data_list):
        super().__init__()
        self.hst_data_list = hst_data_list