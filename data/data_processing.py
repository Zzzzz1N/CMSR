import pandas as pd

import gzip
import json
import torch
import os

from transformers import AutoModel, AutoTokenizer


def list2str(i: list) -> str:
    if len(i) == 0:
        return ""
    else:
        return i[0]


def parse_large_json_gz(file_path: str):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:  # 'rt' 表示以文本模式读取
        for line in f:
            yield json.loads(line.strip())  # 逐行解析为字典


# 批量处理函数
def process_batch(sentences: list, model, tokenizer, device):
    encoded = tokenizer(
        sentences,
        padding=True,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        outputs = model(**encoded)
    return outputs.last_hidden_state[:, 0, :].cpu()


def generate_item_emb(df, model_name, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    batch_size = 16
    embeddings = torch.cat([
        process_batch(df["text_feature"].iloc[i:i + batch_size].tolist(), model, tokenizer, device)
        for i in range(0, len(df), batch_size)
    ], dim=0).numpy()

    output_df = pd.DataFrame({
        "item_id:token": df["asin"],
        "item_emb:float_seq": [' '.join(map(str, emb)) for emb in embeddings]
    })

    output_df.to_csv(output_path, sep='\t', index=False)
    print(f"Embeddings shape: {embeddings.shape}")
    return output_df


if __name__ == "__main__":
    meta_df = pd.DataFrame()
    market_list = ['ca', 'de', 'fr', 'in', 'jp', 'mx', 'uk', 'us']
    output_path = 'data/item_list.csv'
    if os.path.exists(output_path):
        selected_meta_df = pd.read_csv(output_path)
    else:
        chunk_size = 10000  # 根据内存调整分块大小
        chunks = []
        for idx, record in enumerate(parse_large_json_gz("data/meta_Electronics.json.gz")):
            chunks.append(record)
            if (idx + 1) % chunk_size == 0:
                df_chunk = pd.DataFrame(chunks)
                df_chunk = df_chunk[["asin", "title", "description"]]
                meta_df = pd.concat([meta_df, df_chunk], ignore_index=True)
                chunks = []  # 重置临时存储

        # 合并剩余数据
        if chunks:
            df_chunk = pd.DataFrame(chunks)
            df_chunk = df_chunk[["asin", "title", "description"]]
            meta_df = pd.concat([meta_df, df_chunk], ignore_index=True)

        product_set = set().union(
            *(pd.read_csv(f'data/{m}_5core.txt', sep=' ', usecols=['itemId'])['itemId'].unique() for m in market_list))
        selected_meta_df = meta_df[meta_df['asin'].isin(product_set)] \
            .drop_duplicates('asin', keep='first') \
            .assign(description=lambda x: x.description.apply(list2str), \
                    text_feature=lambda x: x['title'] + ' ' + x['description']) \
            .drop(columns=['title', 'description']) \
            .reset_index(drop=True)

        assert selected_meta_df['asin'].nunique() == len(product_set)
        selected_meta_df.to_csv(output_path)

    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    for m in market_list:
        print(f"Generating {m} dataset...")
        if not os.path.exists(f'dataset/{m}'):
            os.mkdir(f'dataset/{m}')
        df = pd.read_csv('data/' + m + '_5core.txt', sep=' ')

        print(f"Generating {m} transactions...")
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df['timestamp:float'] = df.date.values.astype(int)
        df[['userId', 'itemId', 'timestamp:float']].rename(
            columns={'userId': 'user_id:token', 'itemId': 'item_id:token'}).to_csv(f'dataset/{m}/transactions.inter',
                                                                                   sep='\t', index=False)

        # df[['userId', 'itemId']].rename(
        #     columns={'userId': 'user_id', 'itemId': 'item_id'}).to_csv(f'dataset/{m}_transactions.csv', index=False)
        #
        # print(f"Generating {m} user_item_lists...")
        # user_items = (
        #     df.groupby('userId')['itemId']
        #     .apply(set).apply(list)
        #     .reset_index()
        # )
        #
        # user_items[['userId', 'itemId']].rename(
        #     columns={'userId': 'user_id', 'itemId': 'item_list'}).to_csv(f'dataset/{m}_user_item_lists.csv',
        #                                                                  index=False)
        #
        # print(f"Generating {m} item_counts...")
        # item_counts = (
        #     df['itemId'].value_counts()
        #     .reset_index()
        #     .rename(columns={'count': 'item_num', 'itemId': 'item_id'})
        # )
        # item_counts[['item_id', 'item_num']].to_csv(f'dataset/{m}_item_counts.csv', index=False)

        print(f"Generating {m} item_embed...")
        itemset = df.itemId.unique()
        m_seleft_df = selected_meta_df[selected_meta_df['asin'].isin(itemset)]
        m_emb_df = generate_item_emb(m_seleft_df, 'bert-base-uncased', f'dataset/{m}/item_embed.item')
