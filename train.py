from torch_geometric.loader import DataLoader, ImbalancedSampler
from dataset import get_index, GraphDataset, StateTransitionGraph
from model import Trans2Graph
from sklearn import metrics
import dask.dataframe as dd

import torch.nn.functional as F
import torch
import os
import argparse
import numpy as np
import pandas as pd



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='split1', help='split1: 7:1:2, split2: 5:2:3, split3: 3:3:4')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--in_channels', type=int, default=19)
    parser.add_argument('--time_channels', type=int, default=10)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--out_channels', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--use_norm', type=bool, default=True)
    parser.add_argument('--use_time_encoder', type=bool, default=True)
    parser.add_argument('--use_backward', type=bool, default=True)
    parser.add_argument('--use_dropout', type=bool, default=True)
    parser.add_argument('--use_attention', type=bool, default=True)
    parser.add_argument('--use_time_interval', type=bool, default=True)
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--ppr_emb', type=bool, default=True, help='use ppr vector as node embeddings')
    parser.add_argument('--patience', type=int, default=50)
    args = parser.parse_args()
    return args


def data_preparation():
    train_ids = pd.read_csv('train_index/train.csv')
    if not os.path.exists('graph_data/raw/edge_index.pt'):
        all_index = train_ids.index.tolist()
        edge_index = []
        edge_attr = []
        TX_DTYPES = {'value': 'object', 'gas_price': 'object', 'token_address': 'object'}
        train_dfs = dd.read_csv('train_tx.csv', dtype=TX_DTYPES)
        train_dfs = train_dfs.compute()
        train_agg = train_dfs.groupby('address')
        for index in all_index:
            addr = train_ids.iloc[index]['address']
            TransG = StateTransitionGraph(train_agg.get_group(addr), addr)
            [edge_index_, edge_attr_] = TransG.graph_construction(type_ = 'edge_index')
            edge_index.append(edge_index_)
            edge_attr.append(edge_attr_)
        torch.save(edge_index, 'graph_data/raw/edge_index.pt')
        torch.save(edge_attr, 'graph_data/raw/edge_attr.pt')


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_data in train_loader:
        optimizer.zero_grad()
        out = model(batch_data.x.to(device), batch_data.edge_index.to(device), batch_data.edge_attr.to(device), batch_data.batch.to(device))

        loss = F.nll_loss(out, batch_data.y.to(device))
        loss.backward()
        total_loss += loss
        optimizer.step()
    
    loss = total_loss / len(train_loader)
    return loss


@torch.no_grad()
def test(model, data_loader, split_idx, train_ids, device):
    model.eval()
    losses = dict()
    y_preds = {'train': [], 'valid': [], 'test': []}
    
    out = None
    for batch_data in data_loader:
        out_pred = model(batch_data.x.to(device), batch_data.edge_index.to(device), batch_data.edge_attr.to(device), batch_data.batch.to(device))
        if out == None: out = out_pred
        else: out = torch.cat((out, out_pred), dim=0)
        
    for key in split_idx.keys():
        y_preds[key] = out[split_idx[key]].exp()
        losses[key] = F.nll_loss(out[split_idx[key]], torch.tensor(train_ids.iloc[split_idx[key]]['target'].tolist()).to(device)).item()
    
    return losses, y_preds


def train_model():
    metric_values = {'precision': [], 'recall': [], 'micro-f1': [], 'macro-f1': []}
    min_valid_loss = 1e8
    num_round = 0
    split_idx = {'train': train_index, 'valid': val_index, 'test': test_index}
    
    for epoch in range(1, args.epochs + 1):
        loss = train(model, train_loader, optimizer, device)
        losses, out = test(model, data_loader, split_idx, train_ids, device)
        train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']
        
        if valid_loss < min_valid_loss:
            num_round = 0
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_dir + save_model_name)
        else:
            num_round += 1
            
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_loss:.3f}%, '
                        f'Valid: {100 * valid_loss:.3f}% '
                        f'Test: {100 * test_loss:.3f}%')
        if num_round >= args.patience:
            print('{} epoch finished!\n'.format(epoch))
            break


def test_model():
    model.load_state_dict(torch.load(model_dir + save_model_name))
    split_idx = {'train': train_index, 'valid': val_index, 'test': test_index}
    losses, out = test(model, data_loader, split_idx, train_ids, device)
    
    y_val = train_ids.iloc[val_index]['target'].tolist()
    y_pred = np.argmax(out['valid'].tolist(), axis=1)
    print(metrics.classification_report(y_val, y_pred))
    print('Val precision: {}, recall: {}, micro-f1: {}, macro-f1: {}'.format(metrics.precision_score(y_true=y_val, y_pred=y_pred), metrics.recall_score(y_true=y_val, y_pred=y_pred), metrics.f1_score(y_true=y_val, y_pred=y_pred, average='micro'), metrics.f1_score(y_true=y_val, y_pred=y_pred, average='macro')))

    y_test = train_ids.iloc[test_index]['target'].tolist()
    y_test_preds = np.array([class_proba[1] for class_proba in out['test'].cpu()])
    print(f"Average Precisioin Score: {metrics.average_precision_score(y_test, y_test_preds)}")
    print(f"AUC: {metrics.roc_auc_score(y_test, y_test_preds)}")

    y_pred = np.argmax(out['test'].tolist(), axis=1)
    print(metrics.classification_report(y_test,y_pred))
    print('Test precision: {}, recall: {}, micro-f1: {}, macro-f1: {}'.format(metrics.precision_score(y_true=y_test, y_pred=y_pred), metrics.recall_score(y_true=y_test, y_pred=y_pred), metrics.f1_score(y_true=y_test, y_pred=y_pred, average='micro'), metrics.f1_score(y_true=y_test, y_pred=y_pred, average='macro')))

    print(f"Positive recall: {metrics.recall_score(y_true=y_test, y_pred=y_pred, pos_label=1)}")
    print(f"Negative recall: {metrics.recall_score(y_true=y_test, y_pred=y_pred, pos_label=0)}")
    print(f"G-mean: {np.sqrt(metrics.recall_score(y_true=y_test, y_pred=y_pred, pos_label=1) * metrics.recall_score(y_true=y_test, y_pred=y_pred, pos_label=0))}")



if __name__ == "__main__":
    data_preparation()

    args = get_args()
    assert args.time_channels * 2 < args.hidden_channels
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(device)
    
    train_ids = pd.read_csv('./train_index/train.csv')
    train_index, val_index, test_index = get_index(args.split, args.random_seed)
    
    # Dataset and DataLoader for training
    dataset_train = GraphDataset(root='graph_data', train=True, split="split1", ppr_emb=args.ppr_emb)
    sampler = ImbalancedSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler, num_workers=4)

    # Dataset and DataLoader for inference
    dataset_all = GraphDataset(root='graph_data', train=False, ppr_emb=args.ppr_emb)
    data_loader = DataLoader(dataset_all, batch_size=args.batch_size, shuffle=False)
    
    
    # ================ train ================
    model = Trans2Graph(
        in_channels = args.in_channels,
        out_channels = args.out_channels,
        time_channels = args.time_channels,
        hidden_channels = args.hidden_channels,
        num_layers = args.num_layers,
        num_node_types = 19,
        n_heads = args.n_heads,
        use_norm = args.use_norm,
        use_time_encoder = args.use_time_encoder,
        use_backward = args.use_backward,
        use_attention = args.use_attention,
        use_time_interval = args.use_time_interval,
        use_dropout = args.use_dropout,
        p = args.dropout_p,
        ppr_emb = args.ppr_emb
    )
    model.reset_parameters()
    model = model.to(device)
    print(f"Model initialized")
    
    model_dir = "./models/" + args.split + "/trans2graph/"
    save_model_name = "model.pt"
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-7)
    train_model()
    
    # ================ inference ================
    model = Trans2Graph(
        in_channels = args.in_channels,
        out_channels = args.out_channels,
        time_channels = args.time_channels,
        hidden_channels = args.hidden_channels,
        num_layers = args.num_layers,
        num_node_types = 19,
        n_heads = args.n_heads,
        use_norm = args.use_norm,
        use_time_encoder = args.use_time_encoder,
        use_backward = args.use_backward,
        use_attention = args.use_attention,
        use_time_interval = args.use_time_interval,
        use_dropout = False,
        p = args.dropout_p,
        ppr_emb = args.ppr_emb
    )
    model.reset_parameters()
    model = model.to(device)
    test_model()
