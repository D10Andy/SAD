import torch
import torch.utils.data
import os
import numpy as np
from option import args
import random
import pandas as pd
from utils import get_neighbor_finder
from operator import itemgetter

class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)



class DygDataset(torch.utils.data.Dataset):
    def __init__(self, config, split_flag, split_list=None):
        if split_list is None:
            split_list = [0.7, 0.15, 0.15]
        self.config = config

        dataset_name = '{}/ml_{}'.format(self.config.dir_data, self.config.data_set)

        #全量数据 &  标签完整
        self.full_data, self.positive_eids, self.edge_features, self.node_features = \
            self.get_data(dataset_name, split_flag, split_list)
        self.ngh_finder = get_neighbor_finder(self.full_data, uniform=False)
        self.index_start = self.positive_eids[0]

        # 设置训练集中部分无标数据，并使这些无标数据参与训练
        if split_flag == 'train' and self.config.mask_label:
            self.full_data = self.masked_edge_label(self.full_data, self.positive_eids, self.config.mask_ratio)

            # #上采样黑样本
            # upsample_black_example = True
            # if upsample_black_example:
            #     self.full_data,self.positive_eids = self.up_sample_data(self.full_data, self.positive_eids, self.config.mask_ratio)
            #




    def get_data(self, dataset_name, split_flag, split_list):
        graph_df = pd.read_csv('{}.csv'.format(dataset_name))
        edge_features = np.load('{}.npy'.format(dataset_name))
        node_features = np.load('{}_node.npy'.format(dataset_name))

        val_time, test_time = list(np.quantile(graph_df.ts, [split_list[0], split_list[0]+ split_list[1]]))

        sources = graph_df.u.values
        destinations = graph_df.i.values
        edge_idxs = graph_df.idx.values
        labels = graph_df.label.values
        timestamps = graph_df.ts.values

        #random.seed(2020)

        train_mask = np.where(timestamps <= val_time)[0]
        test_mask = np.where(timestamps > test_time)[0]
        val_mask = np.where(np.logical_and(timestamps <= test_time, timestamps > val_time))[0]

        full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

        if split_flag == 'train':
            positive_eids = train_mask
            pass
        elif split_flag == 'valid':
            positive_eids = val_mask
            pass
        elif split_flag == 'test':
            positive_eids = test_mask
            pass
        else:
            # raise RuntimeError(f'no recognize split: {split_flag}')
            positive_eids = np.where(timestamps>=0)


        return full_data, positive_eids, edge_features, node_features

    def masked_edge_label(self, full_data, positive_eids, mask_ratio):
        num = round(len(positive_eids) * mask_ratio)
        mask_label_index = np.random.choice(positive_eids, num, replace=False)
        full_data.labels[mask_label_index] = -1
        return full_data

    def up_sample_data(self, full_data, positive_eids, mask_ratio):
        black_upsample_numbers = np.where(full_data.labels[positive_eids] == 1)[0]
        pickup_index = np.random.choice(black_upsample_numbers, size=int(len(black_upsample_numbers) / (1- mask_ratio) ), replace=True)
        #随机插入的位置
        insert_index = np.random.randint(len(positive_eids), size=len(pickup_index))

        new_edge_idxs = full_data.edge_idxs.copy()
        new_edge_idxs = np.insert(new_edge_idxs,insert_index, new_edge_idxs[pickup_index])

        new_destinations = full_data.destinations.copy()
        new_destinations = np.insert(new_destinations, insert_index, new_destinations[pickup_index])

        new_sources = full_data.sources.copy()
        new_sources = np.insert(new_sources, insert_index, new_sources[pickup_index])

        new_timestamps = full_data.timestamps.copy()
        new_timestamps = np.insert(new_timestamps, insert_index, new_timestamps[pickup_index])

        new_labels = full_data.labels.copy()
        new_labels = np.insert(new_labels, insert_index, new_labels[pickup_index])

        new_full_data = Data(new_sources, new_destinations, new_timestamps, new_edge_idxs, new_labels)

        new_positive_eids = range(len(positive_eids) + len(pickup_index))

        return new_full_data, new_positive_eids


    def edge_padding(self, neigh_edge, neigh_time, edge_feat, src_neigh_idx, source_node):
        neigh_edge = np.concatenate((neigh_edge, np.tile(source_node.reshape(-1, 1), (1, 2))), axis=0)
        neigh_time = np.concatenate((neigh_time, np.zeros([1], dtype=neigh_time.dtype)), axis=0)
        edge_feat = np.concatenate((edge_feat, np.zeros([1, edge_feat.shape[1]], dtype=edge_feat.dtype)), axis=0)
        src_neigh_idx = np.concatenate((src_neigh_idx, np.zeros([1], dtype=src_neigh_idx.dtype)), axis=0)
        return neigh_edge, neigh_time, edge_feat, src_neigh_idx

    def __getitem__(self, item):
        item += self.index_start

        source_node = self.full_data.sources[item]
        current_time = self.full_data.timestamps[item]
        label = self.full_data.labels[item]

        #找邻居
        src_neigh_edge, src_neigh_time, src_neigh_idx = self.ngh_finder.get_temporal_neighbor_all(source_node,
                                                                                                  current_time,
                                                                                                  self.config.n_layer,
                                                                                                  self.config.n_neighbors
                                                                                                  )


        src_edge_feature = self.edge_features[src_neigh_idx].astype(np.float32)
        src_edge_to_time = current_time - src_neigh_time
        src_center_node_idx = np.reshape(source_node, [-1])
        if src_neigh_edge.shape[0] == 0:
            src_neigh_edge, src_edge_to_time, src_edge_feature, src_neigh_idx = self.edge_padding(
                      src_neigh_edge, src_edge_to_time, src_edge_feature, src_neigh_idx, src_center_node_idx)


        label = np.reshape(label, [-1])
        current_time = np.reshape(current_time, [-1])

        return {
            'src_center_node_idx': src_center_node_idx,
            'src_neigh_edge': torch.from_numpy(src_neigh_edge),
            'src_edge_feature': torch.from_numpy(src_edge_feature),
            'src_edge_to_time': torch.from_numpy(src_edge_to_time.astype(np.float32)),
            'init_edge_index': torch.from_numpy(src_neigh_idx),
            'current_time': torch.from_numpy(current_time),
            'label': torch.from_numpy(label)
        }


    def __len__(self):
        return len(self.positive_eids)

class Collate:
    def __init__(self, config):
        self.config = config
        dataset_name = '{}/ml_{}'.format(self.config.dir_data, self.config.data_set)
        self.node_features = np.load('{}_node.npy'.format(dataset_name)).astype(np.float32)

    def reindex_fn(self, edge_list, center_node_idx, batch_idx):

        # 为了对同一个batch内不同子图的数据进行隔离，此处对所有节点数据进行映射
        edge_list_projection = edge_list.view(-1).numpy().tolist()
        edge_list_projection = [str(x) for x in edge_list_projection]

        single_batch_idx = torch.unique(batch_idx).numpy().astype(np.int32).tolist()
        single_batch_idx = [str(x) for x in single_batch_idx]

        batch_idx_projection = batch_idx.reshape([-1, 1]).repeat((1, 2)).view(-1).numpy().astype(np.int32).tolist()
        batch_idx_projection = [str(x) for x in batch_idx_projection]

        center_node_idx_projection = center_node_idx.tolist()
        center_node_idx_projection = [str(x) for x in center_node_idx_projection]

        union_edge_list = list(map(lambda x: x[0] + '_' + x[1], zip(batch_idx_projection, edge_list_projection)))
        union_center_node_list = list(
            map(lambda x: x[0] + '_' + x[1], zip(single_batch_idx, center_node_idx_projection)))

        org_node_id = union_edge_list + union_center_node_list
        org_node_id = list(set(org_node_id))

        new_node_id = torch.arange(0, len(org_node_id)).numpy()
        reid_map = dict(zip(org_node_id, new_node_id))
        true_org_node_id = [int(x.split('_')[1]) for x in org_node_id]
        true_org_node_id = np.array(true_org_node_id)

        keys = union_edge_list
        new_edge_list = itemgetter(*keys)(reid_map)
        new_edge_list = np.array(new_edge_list).reshape([-1, 2])
        new_edge_list = torch.from_numpy(new_edge_list)
        batch_node_features = self.node_features[true_org_node_id]
        new_center_node_idx = np.array(itemgetter(*union_center_node_list)(reid_map))

        return new_center_node_idx, new_edge_list, batch_node_features


    def get_batchidx_fn(self, edge_list):
        batch_size = len(edge_list)
        feat_max_len = np.sum([feat.shape[0] for feat in edge_list])

        mask = torch.zeros((feat_max_len))

        count = 0
        for i, ifeat in enumerate(edge_list):
            size = ifeat.shape[0]
            mask[count:count+size] = i + 1
            count += size
        return mask

    def dyg_collate_fn(self, batch):
        src_edge_feat = torch.cat([b['src_edge_feature'] for b in batch], dim=0)   #n1,f
        src_edge_to_time = torch.cat([b['src_edge_to_time'] for b in batch], dim=0)   #n

        init_edge_index = torch.cat([b['init_edge_index'] for b in batch], dim=0)  # n

        src_center_node_idx = np.concatenate([b['src_center_node_idx'] for b in batch], axis=0) #b

        batch_idx = self.get_batchidx_fn([b['src_neigh_edge'] for b in batch])
        src_neigh_edge = torch.cat([b['src_neigh_edge'] for b in batch], dim=0)  #n,2
        src_center_node_idx, src_neigh_edge, src_node_features = self.reindex_fn(src_neigh_edge, src_center_node_idx, batch_idx)

        label = torch.cat([b['label'] for b in batch], dim=0)
        current_time = torch.cat([b['current_time'] for b in batch], dim=0)

        return {
            'src_edge_feat':src_edge_feat,
            'src_edge_to_time':src_edge_to_time,
            'src_center_node_idx':torch.from_numpy(src_center_node_idx),
            'src_neigh_edge':src_neigh_edge,
            'src_node_features': torch.from_numpy(src_node_features),
            'init_edge_index': init_edge_index,
            'batch_idx': batch_idx,
            'current_time': current_time,
            'labels':label
        }


class RandomDropSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, dataset, drop_rate):
        self.dataset = dataset
        self.drop_rate = drop_rate
        self.drop_num = int(len(dataset) * drop_rate)

    def __iter__(self):
        arange = np.arange(len(self.dataset))
        np.random.shuffle(arange)
        indices = arange[: (1 - self.drop_num)]
        return iter(np.sort(indices))
        # indices = arange
        # return iter(indices)

    def __len__(self):
        return len(self.dataset) - self.drop_num



if __name__ == '__main__':
    config = args
    a = DygDataset(config, 'train')
    #a = DygDatasetTest(config, 'val')
    c = a[5000]
    #print(c)

