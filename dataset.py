from torch_geometric.data import InMemoryDataset, Data
from typing import Callable, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

import torch
import os
import random
import pickle



def get_index(split, random_seed=1):
    """
    process data splits
    """
    if os.path.exists('train_index/train_index_{}.txt'.format(split)):
        train_index = pickle.load(open('train_index/train_index_{}.txt'.format(split), 'rb'))
        val_index = pickle.load(open('train_index/val_index_{}.txt'.format(split), 'rb'))
        test_index = pickle.load(open('train_index/test_index_{}.txt'.format(split), 'rb'))
    else:
        del_index = torch.load('train_index/delete_index.pt')
        index_positive = list(set(train_ids[train_ids['target'] == 1].index.tolist()) - set(del_index))
        index_negative = list(set(train_ids[train_ids['target'] == 0].index.tolist()) - set(del_index))
        random.seed(random_seed)
        random.shuffle(index_positive)
        random.shuffle(index_negative)
        ratio = {'split1': [0.7, 0.8], 'split2': [0.5, 0.7], 'split3': [0.3, 0.6]}
        train_index = list(set(index_positive[:int(len(index_positive) * ratio[split][0])]) | set(index_negative[:int(len(index_negative) * ratio[args.split][0])]))
        val_index = list(set(index_positive[int(len(index_positive) * ratio[split][0]):int(len(index_positive) * ratio[args.split][1])]) | set(index_negative[int(len(index_negative) * ratio[args.split][0]):int(len(index_negative) * ratio[args.split][1])]))
        test_index = list(set(index_positive[int(len(index_positive) * ratio[split][1]):]) | set(index_negative[int(len(index_negative) * ratio[args.split][1]):]))

        pickle.dump(train_index, open('train_index/train_index_{}.txt'.format(split), 'wb'))
        pickle.dump(val_index, open('train_index/val_index_{}.txt'.format(split), 'wb'))
        pickle.dump(test_index, open('train_index/test_index_{}.txt'.format(split), 'wb'))
        
    return train_index, val_index, test_index


def get_ppr_embedding(G: nx.MultiDiGraph, V: set):
    x = np.zeros((19, 19))
    
    for v in V:
        p_dict = nx.pagerank(G, personalization={v: 1.0})
        for u, p in p_dict.items(): x[v][u] = p

    return x


def graph_process(raw_paths, train, train_index=None, ppr_emb=False):
    """
    generate state transition graphs
    """
    data_list = []
    edge_indexs = torch.load(raw_paths[0])
    edge_attrs = torch.load(raw_paths[1])
    train_ids = pd.read_csv('train.csv')

    # process all data
    if not train:
        labels = torch.tensor(train_ids['target'].tolist())
    # process train split
    else:
        labels = torch.tensor(train_ids.iloc[train_index]['target'].tolist())
        edge_indexs = [edge_indexs[idx] for idx in train_index]
        edge_attrs = [edge_attrs[idx] for idx in train_index]
        
    for i in range(len(edge_attrs)):
        if not ppr_emb:
            x = torch.tensor([i for i in range(19)])
            x = x.unsqueeze(1)
            
        else:
            G = nx.MultiDiGraph()
            E = edge_indexs[i]
            V = set(E[0]) | set(E[1])
            
            E_num = len(E[0])
            G.add_edges_from([(E[0][k], E[1][k]) for k in range(E_num)])
            x = get_ppr_embedding(G, V)
            x = torch.tensor(x)
        
        edge_attr = torch.tensor(edge_attrs[i])
        edge_index = torch.tensor(edge_indexs[i])
        # edge_attr contains 5 dimensions: time interval, timestamp of src and trg, node type of src and trg
        edge_attr = torch.cat((edge_attr, edge_index.transpose(-1, -2)), dim=1)

        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            y=labels[i], num_nodes=x.shape[0]
        )
        data_list.append(data)
        
    return data_list


class GraphDataset(InMemoryDataset):
    def __init__(
        self, root: str,
        train: bool = True, split: str = 'split1', ppr_emb: bool = False,
        transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None
    ):
        self.root = root
        self.train = train
        self.ppr_emb = ppr_emb
        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        
        if not self.train:
            path = self.processed_paths[0]
        else:
            path = self.processed_paths[int(split[-1])]
        self.data, self.slices = torch.load(path)
        
    @property
    def raw_file_names(self) -> str:
        return ['edge_index.pt', 'edge_attr.pt']
    
    @property
    def processed_file_names(self) -> List[str]:
        if not self.ppr_emb:
            return ['data_all.pt', 'data_train_split1.pt', 'data_train_split2.pt', 'data_train_split3.pt']
        else:
            return ['data_all_ppr.pt', 'data_train_split1_ppr.pt', 'data_train_split2_ppr.pt', 'data_train_split3_ppr.pt']
    
    def download(self):
        pass
    
    def process(self):
        splits = ['no_train', 'split1', 'split2', 'split3']
        for s in splits:
            if s == "no_train":
                data_list = graph_process(raw_paths=self.raw_paths, train=False, train_index=None, ppr_emb=self.ppr_emb)  # 全部数据
            else:
                train_index, _, _ = get_index(s)
                data_list = graph_process(raw_paths=self.raw_paths, train=True, train_index=train_index, ppr_emb=self.ppr_emb)  # 训练集划分

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            if s == 'no_train':
                torch.save(self.collate(data_list), self.processed_paths[0])
            else:
                torch.save(self.collate(data_list), self.processed_paths[int(s[-1])])


class StateTransitionGraph:
    def __init__(self, df, address):
        self.df = df
        self.address = address
        self.preparation()  # sort the data according to the timestamp

    def preparation(self):
        self.df = self.df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp']).view(int) // 1e9
        self.df = self.df.sort_values(by='timestamp', ascending=True)

    def get_state_token_transfer_(self, row, token, token_balance, states):
        value = float(row['value']) * 1e-18
        if row['from_address'] == self.address:  # transfer out
            balance = token_balance.get(token, 0)
            if balance != 0 and (balance - value) <= 0:
                this_point = states[3]
                token_balance.pop(token)
            elif balance != 0 and (balance - value) > 0:
                this_point = states[2]
                token_balance[token] = balance - value
            else:
                this_point = states[3]
        else:
            if token_balance.get(token, 0) == 0:
                this_point = states[0]
                token_balance[token] = value
            else:
                this_point = states[1]
                token_balance[token] += value
        return this_point

    def get_state_NFT_transfer_(self, row, token, token_balance, states):
        value = (float(row['value']) * 1e-18) if row['edge_type'] == 'ERC1155TokenTransfer' else 1
        if row['from_address'] == self.address:  # tranfer out
            balance = token_balance.get(token, {})
            if balance != {}:
                if len(balance) == 1 and row['token_id'] in balance and (balance[row['token_id']] - value) <= 0:
                    this_point = states[3]
                    token_balance.pop(token)
                elif row['token_id'] in balance and (balance[row['token_id']] - value) > 0:
                    this_point = states[2]
                    token_balance[token][row['token_id']] = balance[row['token_id']] - value
                elif row['token_id'] in balance:
                    this_point = states[2]
                    token_balance[token].pop(row['token_id'])
                else:
                    this_point = states[3]
            else:
                this_point = states[3]
        else:
            if token_balance.get(token, 0) == 0:
                this_point = states[0]
                token_balance[token] = {row['token_id']: value}
            else:
                this_point = states[1]
                if row['edge_type'] == 'ERC1155TokenTransfer' and row['token_id'] in token_balance[token]:
                    token_balance[token][row['token_id']] += value
                else:
                    token_balance[token][row['token_id']] = value
        return this_point

    def graph_construction(self, type_='edge'):
        assert type_ == 'edge' or type_ == 'edge_index'
        last_point = -1
        token_balance = {}
        edges = []
        edge_index = [[], []]
        edge_attr = []
        first_tx_time = int(self.df.iloc[0]['timestamp'])
        for index, row in self.df.iterrows():
            if row['edge_type'] == 'mining_reward':  # mining
                this_point = 0
                self.get_state_token_transfer_(row, 'ETH', token_balance, [5, 6, 7, 8])  # change balance
            elif row['edge_type'] == 'contract_create' and row['from_address'] == self.address:  # create a contract
                this_point = 1
            elif row['edge_type'] == 'contract_create' and row['to_address'] == self.address:  # contract was created
                this_point = 2
            elif (row['edge_type'] == 'contract_call' or row['edge_type'] == 'call_and_transfer' or row[
                'edge_type'] == 'call_and_internal_transfer') and row[
                'from_address'] == self.address:  # invoke a contract
                this_point = 3
                if row['edge_type'] != 'contract_call':
                    if last_point != -1:
                        edges.append(
                            (last_point, this_point, {'from_time': int(last_time), 'to_time': int(row['timestamp'])}))
                        edge_index[0].append(last_point)
                        edge_index[1].append(this_point)
                        edge_attr.append([int(row['timestamp']) - first_tx_time, first_tx_time, int(row['timestamp'])])
                    last_point = this_point
                    last_time = row['timestamp']
                    this_point = self.get_state_token_transfer_(row, 'ETH', token_balance,
                                                                [5, 6, 7, 8])  # change balance
            elif (row['edge_type'] == 'contract_call' or row['edge_type'] == 'call_and_transfer' or row['edge_type'] == 'call_and_internal_transfer') and row['to_address'] == self.address:  # contract was invoked
                this_point = 4
                if row['edge_type'] != 'contract_call':
                    if last_point != -1:
                        edges.append((last_point, this_point,
                                      {'from_time': int(last_time), 'to_time': int(row['timestamp'])}))
                        edge_index[0].append(last_point)
                        edge_index[1].append(this_point)
                        edge_attr.append([int(row['timestamp']) - first_tx_time, first_tx_time, int(row['timestamp'])])
                    last_point = this_point
                    last_time = row['timestamp']
                    this_point = self.get_state_token_transfer_(row, 'ETH', token_balance,
                                                                [5, 6, 7, 8])  # change balance
            elif row['edge_type'] == 'external_transfer' or row['edge_type'] == 'internal_transfer':
                this_point = self.get_state_token_transfer_(row, 'ETH', token_balance, [5, 6, 7, 8])
            elif row['edge_type'] == 'ERC20TokenTransfer':
                this_point = self.get_state_token_transfer_(row, row['token_address'], token_balance, [9, 10, 11, 12])
            elif row['edge_type'] == 'ERC721TokenTransfer' or row['edge_type'] == 'ERC1155TokenTransfer':
                this_point = self.get_state_NFT_transfer_(row, row['token_address'], token_balance, [13, 14, 15, 16])
            elif (row['edge_type'] == 'suicide' or row['edge_type'] == 'suicide and transfer'):
                if row['from_address'] == self.address:
                    this_point = 17
                elif row['to_address'] == self.address:
                    this_point = 18
                if row['edge_type'] != 'suicide':
                    if last_point != -1:
                        edges.append(
                            (last_point, this_point, {'from_time': int(last_time), 'to_time': int(row['timestamp'])}))
                        edge_index[0].append(last_point)
                        edge_index[1].append(this_point)
                        edge_attr.append([int(row['timestamp']) - first_tx_time, first_tx_time, int(row['timestamp'])])
                    last_point = this_point
                    last_time = row['timestamp']
                    this_point = self.get_state_token_transfer_(row, 'ETH', token_balance, [5, 6, 7, 8])
            else:
                continue

            if last_point != -1:
                edges.append((last_point, this_point, {'from_time': int(last_time), 'to_time': int(row['timestamp'])}))
                edge_index[0].append(last_point)
                edge_index[1].append(this_point)
                edge_attr.append([int(row['timestamp']) - first_tx_time, first_tx_time, int(row['timestamp'])])
            last_point = this_point
            last_time = row['timestamp']
        if type_ == 'edge':
            return edges
        else:
            return [edge_index, edge_attr]