from web3._utils.events import get_event_data
# from web3._utils.events import event_abi_to_log_topic
from functools import lru_cache
import traceback
import json
from eth_utils import event_abi_to_log_topic, to_hex
from hexbytes import HexBytes
from web3.auto import w3
import os


# reference: https://docs.alchemy.com/docs/deep-dive-into-eth_getlogs
# abi https://www.smartcontracttoolkit.com/abi

@lru_cache(maxsize=None)
def _get_topic2abi(abi):
    if isinstance(abi, (str)):
        abi = json.loads(abi)

    event_abi = [a for a in abi if a['type'] == 'event']
    topic2abi = {event_abi_to_log_topic(_): _ for _ in event_abi}  # dict: signature: event
    return topic2abi


@lru_cache(maxsize=None)
def _get_hex_topic(t):
    hex_t = HexBytes(t)
    return hex_t


def convert_to_hex(arg, target_schema):
    """
    utility function to convert byte codes into human readable and json serializable data structures
    """
    output = dict()
    for k in arg:
        if isinstance(arg[k], (bytes, bytearray)):
            output[k] = to_hex(arg[k])
        elif isinstance(arg[k], (list)) and len(arg[k]) > 0:
            target = [a for a in target_schema if 'name' in a and a['name'] == k][0]
            if target['type'] == 'tuple[]':
                target_field = target['components']
                output[k] = decode_list_tuple(arg[k], target_field)
            else:
                output[k] = decode_list(arg[k])
        elif isinstance(arg[k], (tuple)):
            target_field = [a['components'] for a in target_schema if 'name' in a and a['name'] == k][0]
            output[k] = decode_tuple(arg[k], target_field)
        else:
            output[k] = arg[k]
    return output


def decode_log(data, topics, abi):
    if abi is not None:
        try:
            topic2abi = _get_topic2abi(abi)
            log = {
                'address': None,  # Web3.toChecksumAddress(address),
                'blockHash': None,  # HexBytes(blockHash),
                'blockNumber': None,
                'data': data,
                'logIndex': None,
                'topics': [_get_hex_topic(_) for _ in topics],
                'transactionHash': None,  # HexBytes(transactionHash),
                'transactionIndex': None
            }
            event_abi = topic2abi[log['topics'][0]]  # signature match
            evt_name = event_abi['name']
            data = get_event_data(w3.codec, event_abi, log)['args']  # Given an event ABI and a log entry for that
            # event, return the decoded event data
            target_schema = event_abi['inputs']
            decoded_data = convert_to_hex(data, target_schema)

            return (evt_name, json.dumps(decoded_data), json.dumps(target_schema))
        except Exception:
            return ('decode error', None, None)
    else:
        return ('no matching abi', None, None)


def main(data_type):
    file_path = 'kaggle_data/forta-protect-web3/tx_log_{}/tx_log_{}/'.format(data_type, data_type)
    for file in os.listdir(file_path):
        with open(file_path + file, 'r') as f:
            file_name = file.split('.')[0]
            with open('kaggle_data/forta-protect-web3/tx_token_{}/'.format(data_type) + file_name + '.csv', 'w') as fw:
                fw.write('blockNumber,timestamp,transactionHash,from_addr,to_addr,edge_type,value,tokenID,tokenAddress,gasUsed,gasPrice\n')
                for line in f:
                    line = line.rstrip('\n')
                    log = json.loads(line)
                    time = log['block_timestamp']
                    block_number = log['block_number']
                    transaction_hash = log['transaction_hash']
                    token_contract = log['address'].lower()
                    pair_abis = {
                        'erc20': '[ { "anonymous": false, "inputs": [ { "indexed": true, "name": "owner", "type": "address" }, { "indexed": true, "name": "spender", "type": "address" }, { "indexed": false, "name": "value", "type": "uint256" } ], "name": "Approval", "type": "event" }, { "anonymous": false, "inputs": [ { "indexed": true, "name": "from", "type": "address" }, { "indexed": true, "name": "to", "type": "address" }, { "indexed": false, "name": "value", "type": "uint256" } ], "name": "Transfer", "type": "event" } ]',
                        'erc721': '[ { "anonymous": false, "inputs": [ { "indexed": true, "internalType": "address", "name": "owner", "type": "address" }, { "indexed": true, "internalType": "address", "name": "approved", "type": "address" }, { "indexed": true, "internalType": "uint256", "name": "tokenId", "type": "uint256" } ], "name": "Approval", "type": "event" }, { "anonymous": false, "inputs": [ { "indexed": true, "internalType": "address", "name": "owner", "type": "address" }, { "indexed": true, "internalType": "address", "name": "operator", "type": "address" }, { "indexed": false, "internalType": "bool", "name": "approved", "type": "bool" } ], "name": "ApprovalForAll", "type": "event" }, { "anonymous": false, "inputs": [ { "indexed": true, "internalType": "address", "name": "from", "type": "address" }, { "indexed": true, "internalType": "address", "name": "to", "type": "address" }, { "indexed": true, "internalType": "uint256", "name": "tokenId", "type": "uint256" } ], "name": "Transfer", "type": "event" } ]',
                        'erc1155': '[ { "anonymous": false, "inputs": [ { "indexed": true, "internalType": "address", "name": "account", "type": "address" }, { "indexed": true, "internalType": "address", "name": "operator", "type": "address" }, { "indexed": false, "internalType": "bool", "name": "approved", "type": "bool" } ], "name": "ApprovalForAll", "type": "event" }, { "anonymous": false, "inputs": [ { "indexed": true, "internalType": "address", "name": "operator", "type": "address" }, { "indexed": true, "internalType": "address", "name": "from", "type": "address" }, { "indexed": true, "internalType": "address", "name": "to", "type": "address" }, { "indexed": false, "internalType": "uint256[]", "name": "ids", "type": "uint256[]" }, { "indexed": false, "internalType": "uint256[]", "name": "values", "type": "uint256[]" } ], "name": "TransferBatch", "type": "event" }, { "anonymous": false, "inputs": [ { "indexed": true, "internalType": "address", "name": "operator", "type": "address" }, { "indexed": true, "internalType": "address", "name": "from", "type": "address" }, { "indexed": true, "internalType": "address", "name": "to", "type": "address" }, { "indexed": false, "internalType": "uint256", "name": "id", "type": "uint256" }, { "indexed": false, "internalType": "uint256", "name": "value", "type": "uint256" } ], "name": "TransferSingle", "type": "event" }, { "anonymous": false, "inputs": [ { "indexed": false, "internalType": "string", "name": "value", "type": "string" }, { "indexed": true, "internalType": "uint256", "name": "id", "type": "uint256" } ], "name": "URI", "type": "event" } ]'
                    }
                    transfer_type_dict = {'erc20': 'ERC20TokenTransfer', 'erc721': 'ERC721TokenTransfer', 'erc1155': 'ERC1155TokenTransfer'}
                    for token_type in pair_abis.keys():
                        output = decode_log(log['data'], log['topics'], pair_abis[token_type])
                        if output[0] == 'Transfer':
                            # fw.write('{};{};{};{};{};{}\n'.format(token_type, block_number, transaction_hash, time, token_contract, output[1]))

                            info_dict = json.loads(output[1])
                            from_addr = info_dict['from']
                            to_addr = info_dict['to']

                            edge_type = transfer_type_dict[token_type]
                            value = ''
                            tokenID = ''
                            if token_type == 'erc20' or token_type == 'erc1155':
                                value = info_dict['value']
                            if token_type == 'erc721' or token_type == 'erc1155':
                                tokenID = info_dict['tokenId']

                            fw.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(block_number, time, transaction_hash, from_addr, to_addr, edge_type, value, tokenID, token_contract, '', ''))


if __name__ == '__main__':
    data_type = 'test' # or train
    main(data_type)
