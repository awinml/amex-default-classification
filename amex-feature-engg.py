import cudf
from utils import generate_aggregate_features

amex_data = cudf.read_parquet("train.parquet")
targets = cudf.read_csv("train_labels.csv")

# Aggregating Features

amex_data["customer_ID"] = (
    amex_data["customer_ID"].str[-16:].str.hex_to_int().astype("int64")
)
amex_data.S_2 = cudf.to_datetime(amex_data.S_2)
amex_data = generate_aggregate_features(amex_data, "customer_ID")


# Converting Cutomer_ID to an Integer Value and merging with the features
targets["customer_ID"] = (
    targets["customer_ID"].str[-16:].str.hex_to_int().astype("int64")
)
targets = targets.set_index("customer_ID")
amex_data = amex_data.merge(targets, left_index=True, right_index=True, how="left")
amex_data.target = amex_data.target.astype("int8")

# Cudf merge randomly shuffles rows
amex_data = amex_data.sort_index().reset_index()
amex_data.drop(["customer_ID"], axis=1, inplace=True)

print("Shape: ", amex_data.shape)
print(f"There are {len(amex_data.columns[1:-1])} features!")

amex_data.to_parquet("amex_data.parquet", index=False)
