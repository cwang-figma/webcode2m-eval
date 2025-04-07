import pandas as pd

df = pd.read_csv("/home/ubuntu/qwen_2_5_vl_outputs_960_short_eval_single_process.csv")

means = df.mean(numeric_only=True)
print("Mean for each numeric column:")
print(means)

cols_to_sum = [" block_match", " text_match", " position_match", " text_color_match", " clip_score"]
selected_means = means[cols_to_sum]

value = 0.2 * selected_means.sum()
print("\n0.2 * sum(block_match, text_match, position_match, text_color_match, clip_score) =", value)
